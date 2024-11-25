"""Core data structures."""

from collections import namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import needle
import numpy
from needle import init

from .backend_numpy import Device, all_devices, cpu

# needle version
LAZY_MODE = False
TENSOR_COUNTER = 0

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api

NDArray = numpy.ndarray

from .backend_selection import NDArray, array_api, default_device


class Op:
    """Operator definition."""

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)


class TensorOp(Op):
    """Op class specialized to output tensors, will be alternate subclasses for other structures"""

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""

    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Optional[Op]
    inputs: List["Value"]
    # The following fields are cached fields for
    # dynamic computation
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


### Not needed in HW1
class TensorTuple(Value):
    """Represent a tuple of tensors.

    To keep things simple, we do not support nested tuples.
    """

    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return needle.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "needle.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else default_device()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=(
                data if not isinstance(data, Tensor) else data.realize_cached_data()
            ),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        # numpy array always sits on cpu
        if array_api is numpy:
            return cpu()
        return data.device

    def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)
    
    # def __getitem__(self, index):
    #     if isinstance(index, tuple):
    #         result = self
    #         # Handle multi-dimensional indexing
    #         idx = [slice(None)] * len(self.shape)  # Default slices for all dimensions
    #         for axis, sub_index in enumerate(index):
    #             if isinstance(sub_index, slice):
    #                 idx[axis] = sub_index
    #             elif sub_index is not None:  # Handle single integer indexing or advanced indexing
    #                 raise ValueError("Only slice indexing is currently supported.")
    #         # Convert slices to start, stop for the appropriate axis
    #         for axis, sub_index in enumerate(idx):
    #             if isinstance(sub_index, slice):
    #                 start, stop, step = sub_index.start, sub_index.stop, sub_index.step
    #                 if step is not None and step != 1:
    #                     raise ValueError("Step size other than 1 is not supported in slicing.")
    #                 # Apply Slice operation for this axis
    #                 result = needle.ops.Slice(start, stop, axis=axis)(result)
    #         return result
    #     elif isinstance(index, slice):
    #         # Handle single-dimension slicing
    #         return needle.ops.Slice(index.start, index.stop, axis=0)(self)
    #     else:
    #         raise TypeError(f"Unsupported index type: {type(index)}")
        
    # def __getitem__(self, index):
    #     if isinstance(index, tuple):
    #         # Handle multi-dimensional indexing
    #         result = self
    #         offset = 0  # Adjust axis indexing when single indices are used
    #         for axis, sub_index in enumerate(index):
    #             if isinstance(sub_index, slice):
    #                 start, stop, step = sub_index.start, sub_index.stop, sub_index.step
    #                 if step is not None and step != 1:
    #                     raise ValueError("Step size other than 1 is not supported in slicing.")
    #                 # Apply Slice operation for the current axis
    #                 result = needle.ops.Slice(start, stop, axis=axis - offset)(result)
    #             elif isinstance(sub_index, int):
    #                 # Integer indexing reduces the dimension, simulate it by slicing and then squeezing
    #                 result = needle.ops.Slice(sub_index, sub_index + 1, axis=axis - offset)(result)
    #                 result = needle.ops.Squeeze(axis=axis - offset)(result)
    #                 offset += 1  # Reduce the subsequent axes by 1
    #             else:
    #                 raise ValueError(f"Unsupported index type: {type(sub_index)}. Only slices and integers are supported.")
    #         return result
    #     elif isinstance(index, slice):
    #         # Handle single-dimension slicing
    #         start, stop, step = index.start, index.stop, index.step
    #         if step is not None and step != 1:
    #             raise ValueError("Step size other than 1 is not supported in slicing.")
    #         return needle.ops.Slice(start, stop, axis=0)(self)
    #     elif isinstance(index, int):
    #         # Handle single-dimension integer indexing
    #         result = needle.ops.Slice(index, index + 1, axis=0)(self)
    #         return needle.ops.Squeeze(axis=0)(result)
    #     else:
    #         raise TypeError(f"Unsupported index type: {type(index)}")

    # def __getitem__(self, index):
    #     if isinstance(index, tuple):
    #         # Handle multi-dimensional indexing
    #         result = self
    #         offset = 0  # Adjust axis indexing when dimensions are reduced
    #         for axis, sub_index in enumerate(index):
    #             if isinstance(sub_index, slice):
    #                 start, stop, step = sub_index.start, sub_index.stop, sub_index.step
    #                 step = step or 1
    #                 if step > 0:
    #                     # Apply StridedSlice operation
    #                     result = needle.ops.StridedSlice(start, stop, step, axis=axis - offset)(result)
    #                 else:
    #                     raise ValueError("Negative step size is not supported.")
    #             elif isinstance(sub_index, int):
    #                 # Integer indexing reduces the dimension, simulate it by slicing and squeezing
    #                 result = needle.ops.Slice(sub_index, sub_index + 1, axis=axis - offset)(result)
    #                 result = needle.ops.Squeeze(axis=axis - offset)(result)
    #                 offset += 1  # Reduce subsequent axes by 1
    #             else:
    #                 raise ValueError(f"Unsupported index type: {type(sub_index)}. Only slices and integers are supported.")
    #         return result
    #     elif isinstance(index, slice):
    #         # Handle single-dimension slicing
    #         start, stop, step = index.start, index.stop, index.step
    #         step = step or 1
    #         if step > 0:
    #             return needle.ops.StridedSlice(start, stop, step, axis=0)(self)
    #         else:
    #             raise ValueError("Negative step size is not supported.")
    #     elif isinstance(index, int):
    #         # Handle single-dimension integer indexing
    #         result = needle.ops.Slice(index, index + 1, axis=0)(self)
    #         return needle.ops.Squeeze(axis=0)(result)
    #     else:
    #         raise TypeError(f"Unsupported index type: {type(index)}")
        
    def __getitem__(self, index):
        def normalize_index(idx, dim_size, axis):
            if isinstance(idx, slice):
                start, stop, step = idx.start, idx.stop, idx.step
                step = step or 1
                if step <= 0:
                    raise ValueError("Negative or zero step size is not supported.")
                
                # Handle None and negative indices
                if start is None:
                    start = 0 if step > 0 else dim_size - 1
                elif start < 0:
                    start = max(0, start + dim_size)
                    
                if stop is None:
                    stop = dim_size if step > 0 else -1
                elif stop < 0:
                    stop = max(0, stop + dim_size)
                    
                return needle.ops.StridedSlice(start, stop, step, axis=axis)
            elif isinstance(idx, int):
                if idx < 0:
                    idx = dim_size + idx
                if not (0 <= idx < dim_size):
                    raise IndexError(f"Index {idx} is out of bounds for axis {axis} with size {dim_size}")
                return idx
            else:
                raise TypeError(f"Unsupported index type: {type(idx)}")

        # Handle single index
        if not isinstance(index, tuple):
            index = (index,)
            
        # Handle Ellipsis
        if Ellipsis in index:
            raise NotImplementedError("Ellipsis indexing is not yet supported")
            
        result = self
        offset = 0  # Adjust for dimension reduction
        
        for axis, idx in enumerate(index):
            if isinstance(idx, slice):
                result = normalize_index(idx, result.shape[axis - offset], axis - offset)(result)
            elif isinstance(idx, int):
                normalized_idx = normalize_index(idx, result.shape[axis - offset], axis - offset)
                result = needle.ops.Slice(normalized_idx, normalized_idx + 1, axis=axis - offset)(result)
                result = needle.ops.Squeeze(axis=axis - offset)(result)
                offset += 1
                
        return result
        
    def __setitem__(self, index, value):
        """
        Set values of the tensor at the specified index/slice.
        
        Parameters:
        -----------
        index : int, slice, or tuple of int/slice
            Index or slice where to set values
        value : scalar or ndarray or Tensor
            Values to set at the specified indices
        """
        if isinstance(value, Tensor):
            value = value.realize_cached_data()
        
        # Ensure we're working with the actual data
        if self.cached_data is None:
            self.realize_cached_data()
            
        # Convert single index/slice to tuple form for unified handling
        if not isinstance(index, tuple):
            index = (index,)
        
        # Handle the assignment using numpy's advanced indexing
        try:
            self.cached_data[index] = value
        except (IndexError, ValueError) as e:
            raise ValueError(f"Invalid indexing or shape mismatch: {str(e)}")
        
    # def __setitem__(self, index, value):
    #     """
    #     Set the values of the tensor at the specified index.
    #     """
    #     breakpoint()
    #     if isinstance(value, Tensor):
    #         value = value.realize_cached_data()
    #     if isinstance(index, tuple):
    #         # Handle multi-dimensional indexing
    #         result = self
    #         offset = 0
    #         for axis, sub_index in enumerate(index):
    #             if isinstance(sub_index, slice):
    #                 start, stop, step = sub_index.start, sub_index.stop, sub_index.step
    #                 step = step or 1
    #                 if step > 0:
    #                     # Apply StridedSlice operation
    #                     result = needle.ops.StridedSlice(start, stop, step, axis=axis - offset)(result)
    #                 else:
    #                     raise ValueError("Negative step size is not supported.")
    #             elif isinstance(sub_index, int):
    #                 # Integer indexing reduces the dimension, simulate it by slicing and squeezing
    #                 result = needle.ops.Slice(sub_index, sub_index + 1, axis=axis - offset)(result)
    #                 result = needle.ops.Squeeze(axis=axis - offset)(result)
    #                 offset += 1
    #             else:
    #                 raise ValueError(f"Unsupported index type: {type(sub_index)}. Only slices and integers are supported.")
    #         # Assign the value to the result tensor
    #         result.cached_data = value
    #         self.cached_data = result.realize_cached_data()
    #     elif isinstance(index, slice):
    #         # Handle single-dimension slicing
    #         start, stop, step = index.start, index.stop, index.step
    #         step = step or 1
    #         if step > 0:
    #             result = needle.ops.StridedSlice(start, stop, step, axis=0)(self)
    #             result.cached_data = value
    #             self.cached_data = result.realize_cached_data()
    #         else:
    #             raise ValueError("Negative step size is not supported.")
    #     elif isinstance(index, int):
    #         # Handle single-dimension integer indexing
    #         result = needle.ops.Slice(index, index + 1, axis=0)(self)
    #         result.cached_data = value
    #         self.cached_data = result.realize_cached_data()
    #     else:
    #         raise TypeError(f"Unsupported index type: {type(index)}")



    def __repr__(self):
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other)
        else:
            return needle.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWisePow()(self, other)
        else:
            return needle.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return needle.ops.MatMul()(self, other)

    def matmul(self, other):
        return needle.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return needle.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return needle.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return needle.ops.Reshape(shape)(self)

    def __neg__(self):
        return needle.ops.Negate()(self)

    def transpose(self, axes=None):
        return needle.ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__


def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))
    for node in reverse_topo_order:
        partial = sum_node_list(node_to_output_grads_list[node])
        node.grad = partial

        if node.is_leaf():
            continue

        for i, grad in zip(node.inputs, node.op.gradient_as_tuple(partial, node)):
            if i not in node_to_output_grads_list:
                node_to_output_grads_list[i] = []
            node_to_output_grads_list[i].append(grad)


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    topo_order = []
    visited = set()

    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if id(node) in visited:
        return

    visited.add(id(node))
    for i in node.inputs:
        topo_sort_dfs(i, visited, topo_order)

    topo_order.append(node)


##############################
####### Helper Methods #######
##############################


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from functools import reduce
    from operator import add

    return reduce(add, node_list)
