с–
ЃБ
:
Add
x"T
y"T
z"T"
Ttype:
2	
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
м
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtypeА
is_initialized
"
dtypetypeШ
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
ц
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И"serve*1.13.12b'v1.13.0-rc2-5-g6612da8'НҐ
|
	input_CNNPlaceholder*
dtype0*/
_output_shapes
:€€€€€€€€€*$
shape:€€€€€€€€€
n
input_DensePlaceholder*
shape:€€€€€€€€€*
dtype0*'
_output_shapes
:€€€€€€€€€
v
conv2d_4/random_uniform/shapeConst*
_output_shapes
:*%
valueB"            *
dtype0
`
conv2d_4/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *№ПЊ
`
conv2d_4/random_uniform/maxConst*
valueB
 *№П>*
dtype0*
_output_shapes
: 
≤
%conv2d_4/random_uniform/RandomUniformRandomUniformconv2d_4/random_uniform/shape*
T0*
dtype0*&
_output_shapes
:*
seed2ЯІњ*
seed±€е)
}
conv2d_4/random_uniform/subSubconv2d_4/random_uniform/maxconv2d_4/random_uniform/min*
T0*
_output_shapes
: 
Ч
conv2d_4/random_uniform/mulMul%conv2d_4/random_uniform/RandomUniformconv2d_4/random_uniform/sub*&
_output_shapes
:*
T0
Й
conv2d_4/random_uniformAddconv2d_4/random_uniform/mulconv2d_4/random_uniform/min*
T0*&
_output_shapes
:
У
conv2d_4/kernel
VariableV2*
shared_name *
dtype0*&
_output_shapes
:*
	container *
shape:
»
conv2d_4/kernel/AssignAssignconv2d_4/kernelconv2d_4/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*&
_output_shapes
:
Ж
conv2d_4/kernel/readIdentityconv2d_4/kernel*
T0*"
_class
loc:@conv2d_4/kernel*&
_output_shapes
:
[
conv2d_4/ConstConst*
_output_shapes
:*
valueB*    *
dtype0
y
conv2d_4/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
≠
conv2d_4/bias/AssignAssignconv2d_4/biasconv2d_4/Const*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(*
_output_shapes
:*
use_locking(
t
conv2d_4/bias/readIdentityconv2d_4/bias*
T0* 
_class
loc:@conv2d_4/bias*
_output_shapes
:
s
"conv2d_4/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
и
conv2d_4/convolutionConv2D	input_CNNconv2d_4/kernel/read*/
_output_shapes
:€€€€€€€€€*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
Ц
conv2d_4/BiasAddBiasAddconv2d_4/convolutionconv2d_4/bias/read*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€*
T0
a
conv2d_4/ReluReluconv2d_4/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€
v
conv2d_5/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv2d_5/random_uniform/minConst*
valueB
 *_¶Њ*
dtype0*
_output_shapes
: 
`
conv2d_5/random_uniform/maxConst*
valueB
 *_¶>*
dtype0*
_output_shapes
: 
≤
%conv2d_5/random_uniform/RandomUniformRandomUniformconv2d_5/random_uniform/shape*
T0*
dtype0*&
_output_shapes
:*
seed2хнС*
seed±€е)
}
conv2d_5/random_uniform/subSubconv2d_5/random_uniform/maxconv2d_5/random_uniform/min*
T0*
_output_shapes
: 
Ч
conv2d_5/random_uniform/mulMul%conv2d_5/random_uniform/RandomUniformconv2d_5/random_uniform/sub*
T0*&
_output_shapes
:
Й
conv2d_5/random_uniformAddconv2d_5/random_uniform/mulconv2d_5/random_uniform/min*
T0*&
_output_shapes
:
У
conv2d_5/kernel
VariableV2*&
_output_shapes
:*
	container *
shape:*
shared_name *
dtype0
»
conv2d_5/kernel/AssignAssignconv2d_5/kernelconv2d_5/random_uniform*
T0*"
_class
loc:@conv2d_5/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
Ж
conv2d_5/kernel/readIdentityconv2d_5/kernel*
T0*"
_class
loc:@conv2d_5/kernel*&
_output_shapes
:
[
conv2d_5/ConstConst*
dtype0*
_output_shapes
:*
valueB*    
y
conv2d_5/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
≠
conv2d_5/bias/AssignAssignconv2d_5/biasconv2d_5/Const*
use_locking(*
T0* 
_class
loc:@conv2d_5/bias*
validate_shape(*
_output_shapes
:
t
conv2d_5/bias/readIdentityconv2d_5/bias*
_output_shapes
:*
T0* 
_class
loc:@conv2d_5/bias
s
"conv2d_5/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
и
conv2d_5/convolutionConv2D	input_CNNconv2d_5/kernel/read*
paddingVALID*/
_output_shapes
:€€€€€€€€€*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
Ц
conv2d_5/BiasAddBiasAddconv2d_5/convolutionconv2d_5/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€
a
conv2d_5/ReluReluconv2d_5/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€
v
conv2d_6/random_uniform/shapeConst*%
valueB"            *
dtype0*
_output_shapes
:
`
conv2d_6/random_uniform/minConst*
valueB
 *_¶Њ*
dtype0*
_output_shapes
: 
`
conv2d_6/random_uniform/maxConst*
valueB
 *_¶>*
dtype0*
_output_shapes
: 
≤
%conv2d_6/random_uniform/RandomUniformRandomUniformconv2d_6/random_uniform/shape*
T0*
dtype0*&
_output_shapes
:*
seed2∞иЅ*
seed±€е)
}
conv2d_6/random_uniform/subSubconv2d_6/random_uniform/maxconv2d_6/random_uniform/min*
_output_shapes
: *
T0
Ч
conv2d_6/random_uniform/mulMul%conv2d_6/random_uniform/RandomUniformconv2d_6/random_uniform/sub*&
_output_shapes
:*
T0
Й
conv2d_6/random_uniformAddconv2d_6/random_uniform/mulconv2d_6/random_uniform/min*&
_output_shapes
:*
T0
У
conv2d_6/kernel
VariableV2*
shape:*
shared_name *
dtype0*&
_output_shapes
:*
	container 
»
conv2d_6/kernel/AssignAssignconv2d_6/kernelconv2d_6/random_uniform*
use_locking(*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:
Ж
conv2d_6/kernel/readIdentityconv2d_6/kernel*&
_output_shapes
:*
T0*"
_class
loc:@conv2d_6/kernel
[
conv2d_6/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
conv2d_6/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
≠
conv2d_6/bias/AssignAssignconv2d_6/biasconv2d_6/Const* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
t
conv2d_6/bias/readIdentityconv2d_6/bias*
T0* 
_class
loc:@conv2d_6/bias*
_output_shapes
:
s
"conv2d_6/convolution/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0
и
conv2d_6/convolutionConv2D	input_CNNconv2d_6/kernel/read*/
_output_shapes
:€€€€€€€€€*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
Ц
conv2d_6/BiasAddBiasAddconv2d_6/convolutionconv2d_6/bias/read*/
_output_shapes
:€€€€€€€€€*
T0*
data_formatNHWC
a
conv2d_6/ReluReluconv2d_6/BiasAdd*/
_output_shapes
:€€€€€€€€€*
T0
\
flatten_4/ShapeShapeconv2d_4/Relu*
_output_shapes
:*
T0*
out_type0
g
flatten_4/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
flatten_4/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
flatten_4/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ѓ
flatten_4/strided_sliceStridedSliceflatten_4/Shapeflatten_4/strided_slice/stackflatten_4/strided_slice/stack_1flatten_4/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0*
shrink_axis_mask 
Y
flatten_4/ConstConst*
valueB: *
dtype0*
_output_shapes
:
~
flatten_4/ProdProdflatten_4/strided_sliceflatten_4/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
\
flatten_4/stack/0Const*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
t
flatten_4/stackPackflatten_4/stack/0flatten_4/Prod*
N*
_output_shapes
:*
T0*

axis 
Е
flatten_4/ReshapeReshapeconv2d_4/Reluflatten_4/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
\
flatten_5/ShapeShapeconv2d_5/Relu*
T0*
out_type0*
_output_shapes
:
g
flatten_5/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
i
flatten_5/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
flatten_5/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ѓ
flatten_5/strided_sliceStridedSliceflatten_5/Shapeflatten_5/strided_slice/stackflatten_5/strided_slice/stack_1flatten_5/strided_slice/stack_2*
end_mask*
_output_shapes
:*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask 
Y
flatten_5/ConstConst*
valueB: *
dtype0*
_output_shapes
:
~
flatten_5/ProdProdflatten_5/strided_sliceflatten_5/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
\
flatten_5/stack/0Const*
valueB :
€€€€€€€€€*
dtype0*
_output_shapes
: 
t
flatten_5/stackPackflatten_5/stack/0flatten_5/Prod*
T0*

axis *
N*
_output_shapes
:
Е
flatten_5/ReshapeReshapeconv2d_5/Reluflatten_5/stack*
T0*
Tshape0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
\
flatten_6/ShapeShapeconv2d_6/Relu*
T0*
out_type0*
_output_shapes
:
g
flatten_6/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
i
flatten_6/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
i
flatten_6/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ѓ
flatten_6/strided_sliceStridedSliceflatten_6/Shapeflatten_6/strided_slice/stackflatten_6/strided_slice/stack_1flatten_6/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0*
shrink_axis_mask 
Y
flatten_6/ConstConst*
valueB: *
dtype0*
_output_shapes
:
~
flatten_6/ProdProdflatten_6/strided_sliceflatten_6/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
\
flatten_6/stack/0Const*
_output_shapes
: *
valueB :
€€€€€€€€€*
dtype0
t
flatten_6/stackPackflatten_6/stack/0flatten_6/Prod*
N*
_output_shapes
:*
T0*

axis 
Е
flatten_6/ReshapeReshapeconv2d_6/Reluflatten_6/stack*0
_output_shapes
:€€€€€€€€€€€€€€€€€€*
T0*
Tshape0
[
concatenate_3/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
ƒ
concatenate_3/concatConcatV2flatten_4/Reshapeflatten_5/Reshapeflatten_6/Reshapeconcatenate_3/concat/axis*

Tidx0*
T0*
N*0
_output_shapes
:€€€€€€€€€€€€€€€€€€
m
dense_9/random_uniform/shapeConst*
valueB"†   @   *
dtype0*
_output_shapes
:
_
dense_9/random_uniform/minConst*
_output_shapes
: *
valueB
 *bЧ'Њ*
dtype0
_
dense_9/random_uniform/maxConst*
_output_shapes
: *
valueB
 *bЧ'>*
dtype0
®
$dense_9/random_uniform/RandomUniformRandomUniformdense_9/random_uniform/shape*
_output_shapes
:	†@*
seed2чК
*
seed±€е)*
T0*
dtype0
z
dense_9/random_uniform/subSubdense_9/random_uniform/maxdense_9/random_uniform/min*
_output_shapes
: *
T0
Н
dense_9/random_uniform/mulMul$dense_9/random_uniform/RandomUniformdense_9/random_uniform/sub*
T0*
_output_shapes
:	†@

dense_9/random_uniformAdddense_9/random_uniform/muldense_9/random_uniform/min*
_output_shapes
:	†@*
T0
Д
dense_9/kernel
VariableV2*
shape:	†@*
shared_name *
dtype0*
_output_shapes
:	†@*
	container 
љ
dense_9/kernel/AssignAssigndense_9/kerneldense_9/random_uniform*
T0*!
_class
loc:@dense_9/kernel*
validate_shape(*
_output_shapes
:	†@*
use_locking(
|
dense_9/kernel/readIdentitydense_9/kernel*
_output_shapes
:	†@*
T0*!
_class
loc:@dense_9/kernel
Z
dense_9/ConstConst*
valueB@*    *
dtype0*
_output_shapes
:@
x
dense_9/bias
VariableV2*
dtype0*
_output_shapes
:@*
	container *
shape:@*
shared_name 
©
dense_9/bias/AssignAssigndense_9/biasdense_9/Const*
use_locking(*
T0*
_class
loc:@dense_9/bias*
validate_shape(*
_output_shapes
:@
q
dense_9/bias/readIdentitydense_9/bias*
T0*
_class
loc:@dense_9/bias*
_output_shapes
:@
Ы
dense_9/MatMulMatMulconcatenate_3/concatdense_9/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€@*
transpose_a( *
transpose_b( 
Ж
dense_9/BiasAddBiasAdddense_9/MatMuldense_9/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€@
W
dense_9/ReluReludense_9/BiasAdd*'
_output_shapes
:€€€€€€€€€@*
T0
n
dense_10/random_uniform/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
`
dense_10/random_uniform/minConst*
_output_shapes
: *
valueB
 *М7МЊ*
dtype0
`
dense_10/random_uniform/maxConst*
valueB
 *М7М>*
dtype0*
_output_shapes
: 
™
%dense_10/random_uniform/RandomUniformRandomUniformdense_10/random_uniform/shape*
T0*
dtype0*
_output_shapes

:@*
seed2уЮѓ*
seed±€е)
}
dense_10/random_uniform/subSubdense_10/random_uniform/maxdense_10/random_uniform/min*
T0*
_output_shapes
: 
П
dense_10/random_uniform/mulMul%dense_10/random_uniform/RandomUniformdense_10/random_uniform/sub*
_output_shapes

:@*
T0
Б
dense_10/random_uniformAdddense_10/random_uniform/muldense_10/random_uniform/min*
T0*
_output_shapes

:@
Г
dense_10/kernel
VariableV2*
dtype0*
_output_shapes

:@*
	container *
shape
:@*
shared_name 
ј
dense_10/kernel/AssignAssigndense_10/kerneldense_10/random_uniform*
T0*"
_class
loc:@dense_10/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
~
dense_10/kernel/readIdentitydense_10/kernel*
_output_shapes

:@*
T0*"
_class
loc:@dense_10/kernel
[
dense_10/ConstConst*
_output_shapes
:*
valueB*    *
dtype0
y
dense_10/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
≠
dense_10/bias/AssignAssigndense_10/biasdense_10/Const*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@dense_10/bias*
validate_shape(
t
dense_10/bias/readIdentitydense_10/bias*
T0* 
_class
loc:@dense_10/bias*
_output_shapes
:
Х
dense_10/MatMulMatMuldense_9/Reludense_10/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
Й
dense_10/BiasAddBiasAdddense_10/MatMuldense_10/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
Y
dense_10/ReluReludense_10/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
n
dense_11/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
`
dense_11/random_uniform/minConst*
valueB
 *±њ*
dtype0*
_output_shapes
: 
`
dense_11/random_uniform/maxConst*
valueB
 *±?*
dtype0*
_output_shapes
: 
™
%dense_11/random_uniform/RandomUniformRandomUniformdense_11/random_uniform/shape*
T0*
dtype0*
_output_shapes

:*
seed2÷ог*
seed±€е)
}
dense_11/random_uniform/subSubdense_11/random_uniform/maxdense_11/random_uniform/min*
T0*
_output_shapes
: 
П
dense_11/random_uniform/mulMul%dense_11/random_uniform/RandomUniformdense_11/random_uniform/sub*
T0*
_output_shapes

:
Б
dense_11/random_uniformAdddense_11/random_uniform/muldense_11/random_uniform/min*
_output_shapes

:*
T0
Г
dense_11/kernel
VariableV2*
_output_shapes

:*
	container *
shape
:*
shared_name *
dtype0
ј
dense_11/kernel/AssignAssigndense_11/kerneldense_11/random_uniform*
use_locking(*
T0*"
_class
loc:@dense_11/kernel*
validate_shape(*
_output_shapes

:
~
dense_11/kernel/readIdentitydense_11/kernel*"
_class
loc:@dense_11/kernel*
_output_shapes

:*
T0
[
dense_11/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
dense_11/bias
VariableV2*
shape:*
shared_name *
dtype0*
_output_shapes
:*
	container 
≠
dense_11/bias/AssignAssigndense_11/biasdense_11/Const*
use_locking(*
T0* 
_class
loc:@dense_11/bias*
validate_shape(*
_output_shapes
:
t
dense_11/bias/readIdentitydense_11/bias* 
_class
loc:@dense_11/bias*
_output_shapes
:*
T0
Ф
dense_11/MatMulMatMulinput_Densedense_11/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
Й
dense_11/BiasAddBiasAdddense_11/MatMuldense_11/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
Y
dense_11/ReluReludense_11/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€
[
concatenate_4/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
†
concatenate_4/concatConcatV2dense_10/Reludense_11/Reluconcatenate_4/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:€€€€€€€€€ 
n
dense_12/random_uniform/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
`
dense_12/random_uniform/minConst*
valueB
 *уµЊ*
dtype0*
_output_shapes
: 
`
dense_12/random_uniform/maxConst*
_output_shapes
: *
valueB
 *уµ>*
dtype0
™
%dense_12/random_uniform/RandomUniformRandomUniformdense_12/random_uniform/shape*
dtype0*
_output_shapes

: *
seed2≤£л*
seed±€е)*
T0
}
dense_12/random_uniform/subSubdense_12/random_uniform/maxdense_12/random_uniform/min*
T0*
_output_shapes
: 
П
dense_12/random_uniform/mulMul%dense_12/random_uniform/RandomUniformdense_12/random_uniform/sub*
T0*
_output_shapes

: 
Б
dense_12/random_uniformAdddense_12/random_uniform/muldense_12/random_uniform/min*
T0*
_output_shapes

: 
Г
dense_12/kernel
VariableV2*
shape
: *
shared_name *
dtype0*
_output_shapes

: *
	container 
ј
dense_12/kernel/AssignAssigndense_12/kerneldense_12/random_uniform*
_output_shapes

: *
use_locking(*
T0*"
_class
loc:@dense_12/kernel*
validate_shape(
~
dense_12/kernel/readIdentitydense_12/kernel*
T0*"
_class
loc:@dense_12/kernel*
_output_shapes

: 
[
dense_12/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
dense_12/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
≠
dense_12/bias/AssignAssigndense_12/biasdense_12/Const*
T0* 
_class
loc:@dense_12/bias*
validate_shape(*
_output_shapes
:*
use_locking(
t
dense_12/bias/readIdentitydense_12/bias*
T0* 
_class
loc:@dense_12/bias*
_output_shapes
:
Э
dense_12/MatMulMatMulconcatenate_4/concatdense_12/kernel/read*
T0*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( 
Й
dense_12/BiasAddBiasAdddense_12/MatMuldense_12/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
Y
dense_12/ReluReludense_12/BiasAdd*'
_output_shapes
:€€€€€€€€€*
T0
n
dense_13/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
`
dense_13/random_uniform/minConst*
valueB
 *0њ*
dtype0*
_output_shapes
: 
`
dense_13/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *0?
™
%dense_13/random_uniform/RandomUniformRandomUniformdense_13/random_uniform/shape*
T0*
dtype0*
_output_shapes

:*
seed2≥Т„*
seed±€е)
}
dense_13/random_uniform/subSubdense_13/random_uniform/maxdense_13/random_uniform/min*
_output_shapes
: *
T0
П
dense_13/random_uniform/mulMul%dense_13/random_uniform/RandomUniformdense_13/random_uniform/sub*
T0*
_output_shapes

:
Б
dense_13/random_uniformAdddense_13/random_uniform/muldense_13/random_uniform/min*
_output_shapes

:*
T0
Г
dense_13/kernel
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
ј
dense_13/kernel/AssignAssigndense_13/kerneldense_13/random_uniform*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@dense_13/kernel
~
dense_13/kernel/readIdentitydense_13/kernel*
T0*"
_class
loc:@dense_13/kernel*
_output_shapes

:
[
dense_13/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
y
dense_13/bias
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
≠
dense_13/bias/AssignAssigndense_13/biasdense_13/Const*
use_locking(*
T0* 
_class
loc:@dense_13/bias*
validate_shape(*
_output_shapes
:
t
dense_13/bias/readIdentitydense_13/bias*
T0* 
_class
loc:@dense_13/bias*
_output_shapes
:
Ц
dense_13/MatMulMatMuldense_12/Reludense_13/kernel/read*'
_output_shapes
:€€€€€€€€€*
transpose_a( *
transpose_b( *
T0
Й
dense_13/BiasAddBiasAdddense_13/MatMuldense_13/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
l
PlaceholderPlaceholder*
dtype0*&
_output_shapes
:*
shape:
ђ
AssignAssignconv2d_4/kernelPlaceholder*&
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(
V
Placeholder_1Placeholder*
shape:*
dtype0*
_output_shapes
:
†
Assign_1Assignconv2d_4/biasPlaceholder_1*
use_locking( *
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(*
_output_shapes
:
n
Placeholder_2Placeholder*&
_output_shapes
:*
shape:*
dtype0
∞
Assign_2Assignconv2d_5/kernelPlaceholder_2*
validate_shape(*&
_output_shapes
:*
use_locking( *
T0*"
_class
loc:@conv2d_5/kernel
V
Placeholder_3Placeholder*
dtype0*
_output_shapes
:*
shape:
†
Assign_3Assignconv2d_5/biasPlaceholder_3*
T0* 
_class
loc:@conv2d_5/bias*
validate_shape(*
_output_shapes
:*
use_locking( 
n
Placeholder_4Placeholder*
shape:*
dtype0*&
_output_shapes
:
∞
Assign_4Assignconv2d_6/kernelPlaceholder_4*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:*
use_locking( *
T0
V
Placeholder_5Placeholder*
dtype0*
_output_shapes
:*
shape:
†
Assign_5Assignconv2d_6/biasPlaceholder_5*
use_locking( *
T0* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes
:
`
Placeholder_6Placeholder*
dtype0*
_output_shapes
:	†@*
shape:	†@
І
Assign_6Assigndense_9/kernelPlaceholder_6*!
_class
loc:@dense_9/kernel*
validate_shape(*
_output_shapes
:	†@*
use_locking( *
T0
V
Placeholder_7Placeholder*
dtype0*
_output_shapes
:@*
shape:@
Ю
Assign_7Assigndense_9/biasPlaceholder_7*
use_locking( *
T0*
_class
loc:@dense_9/bias*
validate_shape(*
_output_shapes
:@
^
Placeholder_8Placeholder*
dtype0*
_output_shapes

:@*
shape
:@
®
Assign_8Assigndense_10/kernelPlaceholder_8*"
_class
loc:@dense_10/kernel*
validate_shape(*
_output_shapes

:@*
use_locking( *
T0
V
Placeholder_9Placeholder*
_output_shapes
:*
shape:*
dtype0
†
Assign_9Assigndense_10/biasPlaceholder_9*
validate_shape(*
_output_shapes
:*
use_locking( *
T0* 
_class
loc:@dense_10/bias
_
Placeholder_10Placeholder*
shape
:*
dtype0*
_output_shapes

:
™
	Assign_10Assigndense_11/kernelPlaceholder_10*
use_locking( *
T0*"
_class
loc:@dense_11/kernel*
validate_shape(*
_output_shapes

:
W
Placeholder_11Placeholder*
dtype0*
_output_shapes
:*
shape:
Ґ
	Assign_11Assigndense_11/biasPlaceholder_11*
T0* 
_class
loc:@dense_11/bias*
validate_shape(*
_output_shapes
:*
use_locking( 
_
Placeholder_12Placeholder*
dtype0*
_output_shapes

: *
shape
: 
™
	Assign_12Assigndense_12/kernelPlaceholder_12*
validate_shape(*
_output_shapes

: *
use_locking( *
T0*"
_class
loc:@dense_12/kernel
W
Placeholder_13Placeholder*
dtype0*
_output_shapes
:*
shape:
Ґ
	Assign_13Assigndense_12/biasPlaceholder_13*
use_locking( *
T0* 
_class
loc:@dense_12/bias*
validate_shape(*
_output_shapes
:
_
Placeholder_14Placeholder*
shape
:*
dtype0*
_output_shapes

:
™
	Assign_14Assigndense_13/kernelPlaceholder_14*
use_locking( *
T0*"
_class
loc:@dense_13/kernel*
validate_shape(*
_output_shapes

:
W
Placeholder_15Placeholder*
_output_shapes
:*
shape:*
dtype0
Ґ
	Assign_15Assigndense_13/biasPlaceholder_15*
_output_shapes
:*
use_locking( *
T0* 
_class
loc:@dense_13/bias*
validate_shape(
И
IsVariableInitializedIsVariableInitializedconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
dtype0*
_output_shapes
: 
Ж
IsVariableInitialized_1IsVariableInitializedconv2d_4/bias*
_output_shapes
: * 
_class
loc:@conv2d_4/bias*
dtype0
К
IsVariableInitialized_2IsVariableInitializedconv2d_5/kernel*"
_class
loc:@conv2d_5/kernel*
dtype0*
_output_shapes
: 
Ж
IsVariableInitialized_3IsVariableInitializedconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
dtype0*
_output_shapes
: 
К
IsVariableInitialized_4IsVariableInitializedconv2d_6/kernel*"
_class
loc:@conv2d_6/kernel*
dtype0*
_output_shapes
: 
Ж
IsVariableInitialized_5IsVariableInitializedconv2d_6/bias* 
_class
loc:@conv2d_6/bias*
dtype0*
_output_shapes
: 
И
IsVariableInitialized_6IsVariableInitializeddense_9/kernel*!
_class
loc:@dense_9/kernel*
dtype0*
_output_shapes
: 
Д
IsVariableInitialized_7IsVariableInitializeddense_9/bias*
_class
loc:@dense_9/bias*
dtype0*
_output_shapes
: 
К
IsVariableInitialized_8IsVariableInitializeddense_10/kernel*"
_class
loc:@dense_10/kernel*
dtype0*
_output_shapes
: 
Ж
IsVariableInitialized_9IsVariableInitializeddense_10/bias* 
_class
loc:@dense_10/bias*
dtype0*
_output_shapes
: 
Л
IsVariableInitialized_10IsVariableInitializeddense_11/kernel*
_output_shapes
: *"
_class
loc:@dense_11/kernel*
dtype0
З
IsVariableInitialized_11IsVariableInitializeddense_11/bias* 
_class
loc:@dense_11/bias*
dtype0*
_output_shapes
: 
Л
IsVariableInitialized_12IsVariableInitializeddense_12/kernel*"
_class
loc:@dense_12/kernel*
dtype0*
_output_shapes
: 
З
IsVariableInitialized_13IsVariableInitializeddense_12/bias* 
_class
loc:@dense_12/bias*
dtype0*
_output_shapes
: 
Л
IsVariableInitialized_14IsVariableInitializeddense_13/kernel*
_output_shapes
: *"
_class
loc:@dense_13/kernel*
dtype0
З
IsVariableInitialized_15IsVariableInitializeddense_13/bias* 
_class
loc:@dense_13/bias*
dtype0*
_output_shapes
: 
К
initNoOp^conv2d_4/bias/Assign^conv2d_4/kernel/Assign^conv2d_5/bias/Assign^conv2d_5/kernel/Assign^conv2d_6/bias/Assign^conv2d_6/kernel/Assign^dense_10/bias/Assign^dense_10/kernel/Assign^dense_11/bias/Assign^dense_11/kernel/Assign^dense_12/bias/Assign^dense_12/kernel/Assign^dense_13/bias/Assign^dense_13/kernel/Assign^dense_9/bias/Assign^dense_9/kernel/Assign
%

group_depsNoOp^dense_13/BiasAdd
G
ConstConst*
value	B : *
dtype0*
_output_shapes
: 

init_all_tablesNoOp
(
legacy_init_opNoOp^init_all_tables
М
init_1NoOp^conv2d_4/bias/Assign^conv2d_4/kernel/Assign^conv2d_5/bias/Assign^conv2d_5/kernel/Assign^conv2d_6/bias/Assign^conv2d_6/kernel/Assign^dense_10/bias/Assign^dense_10/kernel/Assign^dense_11/bias/Assign^dense_11/kernel/Assign^dense_12/bias/Assign^dense_12/kernel/Assign^dense_13/bias/Assign^dense_13/kernel/Assign^dense_9/bias/Assign^dense_9/kernel/Assign

init_2NoOp
&
group_deps_1NoOp^init_1^init_2
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_a2c3041b44794f2ca4a269f2ef806d80/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
М
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
п
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*У
valueЙBЖBconv2d_4/biasBconv2d_4/kernelBconv2d_5/biasBconv2d_5/kernelBconv2d_6/biasBconv2d_6/kernelBdense_10/biasBdense_10/kernelBdense_11/biasBdense_11/kernelBdense_12/biasBdense_12/kernelBdense_13/biasBdense_13/kernelBdense_9/biasBdense_9/kernel*
dtype0
Т
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*3
value*B(B B B B B B B B B B B B B B B B 
Р
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesconv2d_4/biasconv2d_4/kernelconv2d_5/biasconv2d_5/kernelconv2d_6/biasconv2d_6/kerneldense_10/biasdense_10/kerneldense_11/biasdense_11/kerneldense_12/biasdense_12/kerneldense_13/biasdense_13/kerneldense_9/biasdense_9/kernel"/device:CPU:0*
dtypes
2
†
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
ђ
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:
М
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
Й
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
т
save/RestoreV2/tensor_namesConst"/device:CPU:0*У
valueЙBЖBconv2d_4/biasBconv2d_4/kernelBconv2d_5/biasBconv2d_5/kernelBconv2d_6/biasBconv2d_6/kernelBdense_10/biasBdense_10/kernelBdense_11/biasBdense_11/kernelBdense_12/biasBdense_12/kernelBdense_13/biasBdense_13/kernelBdense_9/biasBdense_9/kernel*
dtype0*
_output_shapes
:
Х
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*3
value*B(B B B B B B B B B B B B B B B B 
к
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2
§
save/AssignAssignconv2d_4/biassave/RestoreV2*
use_locking(*
T0* 
_class
loc:@conv2d_4/bias*
validate_shape(*
_output_shapes
:
Є
save/Assign_1Assignconv2d_4/kernelsave/RestoreV2:1*
use_locking(*
T0*"
_class
loc:@conv2d_4/kernel*
validate_shape(*&
_output_shapes
:
®
save/Assign_2Assignconv2d_5/biassave/RestoreV2:2* 
_class
loc:@conv2d_5/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Є
save/Assign_3Assignconv2d_5/kernelsave/RestoreV2:3*"
_class
loc:@conv2d_5/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0
®
save/Assign_4Assignconv2d_6/biassave/RestoreV2:4* 
_class
loc:@conv2d_6/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
Є
save/Assign_5Assignconv2d_6/kernelsave/RestoreV2:5*
T0*"
_class
loc:@conv2d_6/kernel*
validate_shape(*&
_output_shapes
:*
use_locking(
®
save/Assign_6Assigndense_10/biassave/RestoreV2:6*
T0* 
_class
loc:@dense_10/bias*
validate_shape(*
_output_shapes
:*
use_locking(
∞
save/Assign_7Assigndense_10/kernelsave/RestoreV2:7*
_output_shapes

:@*
use_locking(*
T0*"
_class
loc:@dense_10/kernel*
validate_shape(
®
save/Assign_8Assigndense_11/biassave/RestoreV2:8*
use_locking(*
T0* 
_class
loc:@dense_11/bias*
validate_shape(*
_output_shapes
:
∞
save/Assign_9Assigndense_11/kernelsave/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@dense_11/kernel*
validate_shape(*
_output_shapes

:
™
save/Assign_10Assigndense_12/biassave/RestoreV2:10*
T0* 
_class
loc:@dense_12/bias*
validate_shape(*
_output_shapes
:*
use_locking(
≤
save/Assign_11Assigndense_12/kernelsave/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@dense_12/kernel*
validate_shape(*
_output_shapes

: 
™
save/Assign_12Assigndense_13/biassave/RestoreV2:12*
use_locking(*
T0* 
_class
loc:@dense_13/bias*
validate_shape(*
_output_shapes
:
≤
save/Assign_13Assigndense_13/kernelsave/RestoreV2:13*
_output_shapes

:*
use_locking(*
T0*"
_class
loc:@dense_13/kernel*
validate_shape(
®
save/Assign_14Assigndense_9/biassave/RestoreV2:14*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@dense_9/bias
±
save/Assign_15Assigndense_9/kernelsave/RestoreV2:15*
use_locking(*
T0*!
_class
loc:@dense_9/kernel*
validate_shape(*
_output_shapes
:	†@
Ю
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8")
saved_model_main_op

legacy_init_op"±
	variables£†
`
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02conv2d_4/random_uniform:08
Q
conv2d_4/bias:0conv2d_4/bias/Assignconv2d_4/bias/read:02conv2d_4/Const:08
`
conv2d_5/kernel:0conv2d_5/kernel/Assignconv2d_5/kernel/read:02conv2d_5/random_uniform:08
Q
conv2d_5/bias:0conv2d_5/bias/Assignconv2d_5/bias/read:02conv2d_5/Const:08
`
conv2d_6/kernel:0conv2d_6/kernel/Assignconv2d_6/kernel/read:02conv2d_6/random_uniform:08
Q
conv2d_6/bias:0conv2d_6/bias/Assignconv2d_6/bias/read:02conv2d_6/Const:08
\
dense_9/kernel:0dense_9/kernel/Assigndense_9/kernel/read:02dense_9/random_uniform:08
M
dense_9/bias:0dense_9/bias/Assigndense_9/bias/read:02dense_9/Const:08
`
dense_10/kernel:0dense_10/kernel/Assigndense_10/kernel/read:02dense_10/random_uniform:08
Q
dense_10/bias:0dense_10/bias/Assigndense_10/bias/read:02dense_10/Const:08
`
dense_11/kernel:0dense_11/kernel/Assigndense_11/kernel/read:02dense_11/random_uniform:08
Q
dense_11/bias:0dense_11/bias/Assigndense_11/bias/read:02dense_11/Const:08
`
dense_12/kernel:0dense_12/kernel/Assigndense_12/kernel/read:02dense_12/random_uniform:08
Q
dense_12/bias:0dense_12/bias/Assigndense_12/bias/read:02dense_12/Const:08
`
dense_13/kernel:0dense_13/kernel/Assigndense_13/kernel/read:02dense_13/random_uniform:08
Q
dense_13/bias:0dense_13/bias/Assigndense_13/bias/read:02dense_13/Const:08"ї
trainable_variables£†
`
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02conv2d_4/random_uniform:08
Q
conv2d_4/bias:0conv2d_4/bias/Assignconv2d_4/bias/read:02conv2d_4/Const:08
`
conv2d_5/kernel:0conv2d_5/kernel/Assignconv2d_5/kernel/read:02conv2d_5/random_uniform:08
Q
conv2d_5/bias:0conv2d_5/bias/Assignconv2d_5/bias/read:02conv2d_5/Const:08
`
conv2d_6/kernel:0conv2d_6/kernel/Assignconv2d_6/kernel/read:02conv2d_6/random_uniform:08
Q
conv2d_6/bias:0conv2d_6/bias/Assignconv2d_6/bias/read:02conv2d_6/Const:08
\
dense_9/kernel:0dense_9/kernel/Assigndense_9/kernel/read:02dense_9/random_uniform:08
M
dense_9/bias:0dense_9/bias/Assigndense_9/bias/read:02dense_9/Const:08
`
dense_10/kernel:0dense_10/kernel/Assigndense_10/kernel/read:02dense_10/random_uniform:08
Q
dense_10/bias:0dense_10/bias/Assigndense_10/bias/read:02dense_10/Const:08
`
dense_11/kernel:0dense_11/kernel/Assigndense_11/kernel/read:02dense_11/random_uniform:08
Q
dense_11/bias:0dense_11/bias/Assigndense_11/bias/read:02dense_11/Const:08
`
dense_12/kernel:0dense_12/kernel/Assigndense_12/kernel/read:02dense_12/random_uniform:08
Q
dense_12/bias:0dense_12/bias/Assigndense_12/bias/read:02dense_12/Const:08
`
dense_13/kernel:0dense_13/kernel/Assigndense_13/kernel/read:02dense_13/random_uniform:08
Q
dense_13/bias:0dense_13/bias/Assigndense_13/bias/read:02dense_13/Const:08*—
serving_defaultљ
5
input_0*
input_CNN:0€€€€€€€€€
/
input_1$
input_Dense:0€€€€€€€€€7

prediction)
dense_13/BiasAdd:0€€€€€€€€€tensorflow/serving/predict