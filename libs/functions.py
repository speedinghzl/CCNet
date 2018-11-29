import torch.autograd as autograd
import torch.cuda.comm as comm
from torch.autograd.function import once_differentiable

from . import _ext

# Activation names
ACT_LEAKY_RELU = "leaky_relu"
ACT_ELU = "elu"
ACT_NONE = "none"


def _check(fn, *args, **kwargs):
    success = fn(*args, **kwargs)
    if not success:
        raise RuntimeError("CUDA Error encountered in {}".format(fn))


def _broadcast_shape(x):
    out_size = []
    for i, s in enumerate(x.size()):
        if i != 1:
            out_size.append(1)
        else:
            out_size.append(s)
    return out_size


def _reduce(x):
    if len(x.size()) == 2:
        return x.sum(dim=0)
    else:
        n, c = x.size()[0:2]
        return x.contiguous().view((n, c, -1)).sum(2).sum(0)


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


def _act_forward(ctx, x):
    if ctx.activation == ACT_LEAKY_RELU:
        _check(_ext.leaky_relu_cuda, x, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _check(_ext.elu_cuda, x)
    elif ctx.activation == ACT_NONE:
        pass


def _act_backward(ctx, x, dx):
    if ctx.activation == ACT_LEAKY_RELU:
        _check(_ext.leaky_relu_backward_cuda, x, dx, ctx.slope)
        _check(_ext.leaky_relu_cuda, x, 1. / ctx.slope)
    elif ctx.activation == ACT_ELU:
        _check(_ext.elu_backward_cuda, x, dx)
        _check(_ext.elu_inv_cuda, x)
    elif ctx.activation == ACT_NONE:
        pass


def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")


class InPlaceABN(autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var,
                training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
        # Save context
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope

        n = _count_samples(x)

        if ctx.training:
            mean = x.new().resize_as_(running_mean)
            var = x.new().resize_as_(running_var)
            _check_contiguous(x, mean, var)
            _check(_ext.bn_mean_var_cuda, x, mean, var)

            # Update running stats
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * var * n / (n - 1))
        else:
            mean, var = running_mean, running_var

        _check_contiguous(x, mean, var, weight, bias)
        _check(_ext.bn_forward_cuda,
               x, mean, var,
               weight if weight is not None else x.new(),
               bias if bias is not None else x.new(),
               x, x, ctx.eps)

        # Activation
        _act_forward(ctx, x)

        # Output
        ctx.var = var
        ctx.save_for_backward(x, weight, bias, running_mean, running_var)
        ctx.mark_dirty(x)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, weight, bias, running_mean, running_var = ctx.saved_tensors
        dz = dz.contiguous()

        # Undo activation
        _act_backward(ctx, z, dz)

        if ctx.needs_input_grad[0]:
            dx = dz.new().resize_as_(dz)
        else:
            dx = None

        if ctx.needs_input_grad[1]:
            dweight = dz.new().resize_as_(running_mean).zero_()
        else:
            dweight = None

        if ctx.needs_input_grad[2]:
            dbias = dz.new().resize_as_(running_mean).zero_()
        else:
            dbias = None

        if ctx.training:
            edz = dz.new().resize_as_(running_mean)
            eydz = dz.new().resize_as_(running_mean)
            _check_contiguous(z, dz, weight, bias, edz, eydz)
            _check(_ext.bn_edz_eydz_cuda,
                   z, dz,
                   weight if weight is not None else dz.new(),
                   bias if bias is not None else dz.new(),
                   edz, eydz, ctx.eps)
        else:
            # TODO: implement CUDA backward for inference mode
            edz = dz.new().resize_as_(running_mean).zero_()
            eydz = dz.new().resize_as_(running_mean).zero_()

        _check_contiguous(dz, z, ctx.var, weight, bias, edz, eydz, dx, dweight, dbias)
        _check(_ext.bn_backard_cuda,
               dz, z, ctx.var,
               weight if weight is not None else dz.new(),
               bias if bias is not None else dz.new(),
               edz, eydz,
               dx if dx is not None else dz.new(),
               dweight if dweight is not None else dz.new(),
               dbias if dbias is not None else dz.new(),
               ctx.eps)

        del ctx.var

        return dx, dweight, dbias, None, None, None, None, None, None, None


class InPlaceABNSync(autograd.Function):
    @classmethod
    def forward(cls, ctx, x, weight, bias, running_mean, running_var,
                extra, training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
        # Save context
        cls._parse_extra(ctx, extra)
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope

        n = _count_samples(x) * (ctx.master_queue.maxsize + 1)

        if ctx.training:
            mean = x.new().resize_(1, running_mean.size(0))
            var = x.new().resize_(1, running_var.size(0))
            _check_contiguous(x, mean, var)
            _check(_ext.bn_mean_var_cuda, x, mean, var)

            if ctx.is_master:
                means, vars = [mean], [var]
                for _ in range(ctx.master_queue.maxsize):
                    mean_w, var_w = ctx.master_queue.get()
                    ctx.master_queue.task_done()
                    means.append(mean_w)
                    vars.append(var_w)

                means = comm.gather(means)
                vars = comm.gather(vars)

                mean = means.mean(0)
                var = (vars + (mean - means) ** 2).mean(0)

                tensors = comm.broadcast_coalesced((mean, var), [mean.get_device()] + ctx.worker_ids)
                for ts, queue in zip(tensors[1:], ctx.worker_queues):
                    queue.put(ts)
            else:
                ctx.master_queue.put((mean, var))
                mean, var = ctx.worker_queue.get()
                ctx.worker_queue.task_done()

            # Update running stats
            running_mean.mul_((1 - ctx.momentum)).add_(ctx.momentum * mean)
            running_var.mul_((1 - ctx.momentum)).add_(ctx.momentum * var * n / (n - 1))
        else:
            mean, var = running_mean, running_var

        _check_contiguous(x, mean, var, weight, bias)
        _check(_ext.bn_forward_cuda,
               x, mean, var,
               weight if weight is not None else x.new(),
               bias if bias is not None else x.new(),
               x, x, ctx.eps)

        # Activation
        _act_forward(ctx, x)

        # Output
        ctx.var = var
        ctx.save_for_backward(x, weight, bias, running_mean, running_var)
        ctx.mark_dirty(x)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, weight, bias, running_mean, running_var = ctx.saved_tensors
        dz = dz.contiguous()

        # Undo activation
        _act_backward(ctx, z, dz)

        if ctx.needs_input_grad[0]:
            dx = dz.new().resize_as_(dz)
        else:
            dx = None

        if ctx.needs_input_grad[1]:
            dweight = dz.new().resize_as_(running_mean).zero_()
        else:
            dweight = None

        if ctx.needs_input_grad[2]:
            dbias = dz.new().resize_as_(running_mean).zero_()
        else:
            dbias = None

        if ctx.training:
            edz = dz.new().resize_as_(running_mean)
            eydz = dz.new().resize_as_(running_mean)
            _check_contiguous(z, dz, weight, bias, edz, eydz)
            _check(_ext.bn_edz_eydz_cuda,
                   z, dz,
                   weight if weight is not None else dz.new(),
                   bias if bias is not None else dz.new(),
                   edz, eydz, ctx.eps)

            if ctx.is_master:
                edzs, eydzs = [edz], [eydz]
                for _ in range(len(ctx.worker_queues)):
                    edz_w, eydz_w = ctx.master_queue.get()
                    ctx.master_queue.task_done()
                    edzs.append(edz_w)
                    eydzs.append(eydz_w)

                edz = comm.reduce_add(edzs) / (ctx.master_queue.maxsize + 1)
                eydz = comm.reduce_add(eydzs) / (ctx.master_queue.maxsize + 1)

                tensors = comm.broadcast_coalesced((edz, eydz), [edz.get_device()] + ctx.worker_ids)
                for ts, queue in zip(tensors[1:], ctx.worker_queues):
                    queue.put(ts)
            else:
                ctx.master_queue.put((edz, eydz))
                edz, eydz = ctx.worker_queue.get()
                ctx.worker_queue.task_done()
        else:
            edz = dz.new().resize_as_(running_mean).zero_()
            eydz = dz.new().resize_as_(running_mean).zero_()

        _check_contiguous(dz, z, ctx.var, weight, bias, edz, eydz, dx, dweight, dbias)
        _check(_ext.bn_backard_cuda,
               dz, z, ctx.var,
               weight if weight is not None else dz.new(),
               bias if bias is not None else dz.new(),
               edz, eydz,
               dx if dx is not None else dz.new(),
               dweight if dweight is not None else dz.new(),
               dbias if dbias is not None else dz.new(),
               ctx.eps)

        del ctx.var

        return dx, dweight, dbias, None, None, None, None, None, None, None, None

    @staticmethod
    def _parse_extra(ctx, extra):
        ctx.is_master = extra["is_master"]
        if ctx.is_master:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queues = extra["worker_queues"]
            ctx.worker_ids = extra["worker_ids"]
        else:
            ctx.master_queue = extra["master_queue"]
            ctx.worker_queue = extra["worker_queue"]


inplace_abn = InPlaceABN.apply
inplace_abn_sync = InPlaceABNSync.apply

__all__ = ["inplace_abn", "inplace_abn_sync"]
