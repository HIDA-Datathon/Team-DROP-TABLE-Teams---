щО!
Щэ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.1.02unknown8ШД
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
Ѓ
%sequential_2/sequential/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%sequential_2/sequential/conv2d/kernel
І
9sequential_2/sequential/conv2d/kernel/Read/ReadVariableOpReadVariableOp%sequential_2/sequential/conv2d/kernel*&
_output_shapes
:@*
dtype0
Ю
#sequential_2/sequential/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#sequential_2/sequential/conv2d/bias
Ч
7sequential_2/sequential/conv2d/bias/Read/ReadVariableOpReadVariableOp#sequential_2/sequential/conv2d/bias*
_output_shapes
:@*
dtype0
≤
'sequential_2/sequential/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*8
shared_name)'sequential_2/sequential/conv2d_1/kernel
Ђ
;sequential_2/sequential/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp'sequential_2/sequential/conv2d_1/kernel*&
_output_shapes
:@@*
dtype0
Ґ
%sequential_2/sequential/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%sequential_2/sequential/conv2d_1/bias
Ы
9sequential_2/sequential/conv2d_1/bias/Read/ReadVariableOpReadVariableOp%sequential_2/sequential/conv2d_1/bias*
_output_shapes
:@*
dtype0
≤
'sequential_2/sequential/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*8
shared_name)'sequential_2/sequential/conv2d_2/kernel
Ђ
;sequential_2/sequential/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp'sequential_2/sequential/conv2d_2/kernel*&
_output_shapes
:@@*
dtype0
Ґ
%sequential_2/sequential/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%sequential_2/sequential/conv2d_2/bias
Ы
9sequential_2/sequential/conv2d_2/bias/Read/ReadVariableOpReadVariableOp%sequential_2/sequential/conv2d_2/bias*
_output_shapes
:@*
dtype0
≥
'sequential_2/sequential/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*8
shared_name)'sequential_2/sequential/conv2d_3/kernel
ђ
;sequential_2/sequential/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp'sequential_2/sequential/conv2d_3/kernel*'
_output_shapes
:@А*
dtype0
£
%sequential_2/sequential/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%sequential_2/sequential/conv2d_3/bias
Ь
9sequential_2/sequential/conv2d_3/bias/Read/ReadVariableOpReadVariableOp%sequential_2/sequential/conv2d_3/bias*
_output_shapes	
:А*
dtype0
•
$sequential_2/sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*5
shared_name&$sequential_2/sequential/dense/kernel
Ю
8sequential_2/sequential/dense/kernel/Read/ReadVariableOpReadVariableOp$sequential_2/sequential/dense/kernel*
_output_shapes
:	А*
dtype0
Ь
"sequential_2/sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"sequential_2/sequential/dense/bias
Х
6sequential_2/sequential/dense/bias/Read/ReadVariableOpReadVariableOp"sequential_2/sequential/dense/bias*
_output_shapes
:*
dtype0
≠
(sequential_2/sequential_1/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*9
shared_name*(sequential_2/sequential_1/dense_1/kernel
¶
<sequential_2/sequential_1/dense_1/kernel/Read/ReadVariableOpReadVariableOp(sequential_2/sequential_1/dense_1/kernel*
_output_shapes
:	А*
dtype0
•
&sequential_2/sequential_1/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&sequential_2/sequential_1/dense_1/bias
Ю
:sequential_2/sequential_1/dense_1/bias/Read/ReadVariableOpReadVariableOp&sequential_2/sequential_1/dense_1/bias*
_output_shapes	
:А*
dtype0
«
1sequential_2/sequential_1/conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*B
shared_name31sequential_2/sequential_1/conv2d_transpose/kernel
ј
Esequential_2/sequential_1/conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOp1sequential_2/sequential_1/conv2d_transpose/kernel*'
_output_shapes
:@А*
dtype0
ґ
/sequential_2/sequential_1/conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/sequential_2/sequential_1/conv2d_transpose/bias
ѓ
Csequential_2/sequential_1/conv2d_transpose/bias/Read/ReadVariableOpReadVariableOp/sequential_2/sequential_1/conv2d_transpose/bias*
_output_shapes
:@*
dtype0
 
3sequential_2/sequential_1/conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*D
shared_name53sequential_2/sequential_1/conv2d_transpose_1/kernel
√
Gsequential_2/sequential_1/conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOp3sequential_2/sequential_1/conv2d_transpose_1/kernel*&
_output_shapes
:@@*
dtype0
Ї
1sequential_2/sequential_1/conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31sequential_2/sequential_1/conv2d_transpose_1/bias
≥
Esequential_2/sequential_1/conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOp1sequential_2/sequential_1/conv2d_transpose_1/bias*
_output_shapes
:@*
dtype0
 
3sequential_2/sequential_1/conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*D
shared_name53sequential_2/sequential_1/conv2d_transpose_2/kernel
√
Gsequential_2/sequential_1/conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOp3sequential_2/sequential_1/conv2d_transpose_2/kernel*&
_output_shapes
:@@*
dtype0
Ї
1sequential_2/sequential_1/conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31sequential_2/sequential_1/conv2d_transpose_2/bias
≥
Esequential_2/sequential_1/conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOp1sequential_2/sequential_1/conv2d_transpose_2/bias*
_output_shapes
:@*
dtype0
 
3sequential_2/sequential_1/conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*D
shared_name53sequential_2/sequential_1/conv2d_transpose_3/kernel
√
Gsequential_2/sequential_1/conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOp3sequential_2/sequential_1/conv2d_transpose_3/kernel*&
_output_shapes
:@*
dtype0
Ї
1sequential_2/sequential_1/conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31sequential_2/sequential_1/conv2d_transpose_3/bias
≥
Esequential_2/sequential_1/conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOp1sequential_2/sequential_1/conv2d_transpose_3/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
Љ
,Adam/sequential_2/sequential/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,Adam/sequential_2/sequential/conv2d/kernel/m
µ
@Adam/sequential_2/sequential/conv2d/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/sequential_2/sequential/conv2d/kernel/m*&
_output_shapes
:@*
dtype0
ђ
*Adam/sequential_2/sequential/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/sequential_2/sequential/conv2d/bias/m
•
>Adam/sequential_2/sequential/conv2d/bias/m/Read/ReadVariableOpReadVariableOp*Adam/sequential_2/sequential/conv2d/bias/m*
_output_shapes
:@*
dtype0
ј
.Adam/sequential_2/sequential/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*?
shared_name0.Adam/sequential_2/sequential/conv2d_1/kernel/m
є
BAdam/sequential_2/sequential/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/sequential_2/sequential/conv2d_1/kernel/m*&
_output_shapes
:@@*
dtype0
∞
,Adam/sequential_2/sequential/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,Adam/sequential_2/sequential/conv2d_1/bias/m
©
@Adam/sequential_2/sequential/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOp,Adam/sequential_2/sequential/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
ј
.Adam/sequential_2/sequential/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*?
shared_name0.Adam/sequential_2/sequential/conv2d_2/kernel/m
є
BAdam/sequential_2/sequential/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/sequential_2/sequential/conv2d_2/kernel/m*&
_output_shapes
:@@*
dtype0
∞
,Adam/sequential_2/sequential/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,Adam/sequential_2/sequential/conv2d_2/bias/m
©
@Adam/sequential_2/sequential/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOp,Adam/sequential_2/sequential/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
Ѕ
.Adam/sequential_2/sequential/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*?
shared_name0.Adam/sequential_2/sequential/conv2d_3/kernel/m
Ї
BAdam/sequential_2/sequential/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/sequential_2/sequential/conv2d_3/kernel/m*'
_output_shapes
:@А*
dtype0
±
,Adam/sequential_2/sequential/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*=
shared_name.,Adam/sequential_2/sequential/conv2d_3/bias/m
™
@Adam/sequential_2/sequential/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOp,Adam/sequential_2/sequential/conv2d_3/bias/m*
_output_shapes	
:А*
dtype0
≥
+Adam/sequential_2/sequential/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*<
shared_name-+Adam/sequential_2/sequential/dense/kernel/m
ђ
?Adam/sequential_2/sequential/dense/kernel/m/Read/ReadVariableOpReadVariableOp+Adam/sequential_2/sequential/dense/kernel/m*
_output_shapes
:	А*
dtype0
™
)Adam/sequential_2/sequential/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/sequential_2/sequential/dense/bias/m
£
=Adam/sequential_2/sequential/dense/bias/m/Read/ReadVariableOpReadVariableOp)Adam/sequential_2/sequential/dense/bias/m*
_output_shapes
:*
dtype0
ї
/Adam/sequential_2/sequential_1/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*@
shared_name1/Adam/sequential_2/sequential_1/dense_1/kernel/m
і
CAdam/sequential_2/sequential_1/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp/Adam/sequential_2/sequential_1/dense_1/kernel/m*
_output_shapes
:	А*
dtype0
≥
-Adam/sequential_2/sequential_1/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*>
shared_name/-Adam/sequential_2/sequential_1/dense_1/bias/m
ђ
AAdam/sequential_2/sequential_1/dense_1/bias/m/Read/ReadVariableOpReadVariableOp-Adam/sequential_2/sequential_1/dense_1/bias/m*
_output_shapes	
:А*
dtype0
’
8Adam/sequential_2/sequential_1/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*I
shared_name:8Adam/sequential_2/sequential_1/conv2d_transpose/kernel/m
ќ
LAdam/sequential_2/sequential_1/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOp8Adam/sequential_2/sequential_1/conv2d_transpose/kernel/m*'
_output_shapes
:@А*
dtype0
ƒ
6Adam/sequential_2/sequential_1/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*G
shared_name86Adam/sequential_2/sequential_1/conv2d_transpose/bias/m
љ
JAdam/sequential_2/sequential_1/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOp6Adam/sequential_2/sequential_1/conv2d_transpose/bias/m*
_output_shapes
:@*
dtype0
Ў
:Adam/sequential_2/sequential_1/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*K
shared_name<:Adam/sequential_2/sequential_1/conv2d_transpose_1/kernel/m
—
NAdam/sequential_2/sequential_1/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp:Adam/sequential_2/sequential_1/conv2d_transpose_1/kernel/m*&
_output_shapes
:@@*
dtype0
»
8Adam/sequential_2/sequential_1/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adam/sequential_2/sequential_1/conv2d_transpose_1/bias/m
Ѕ
LAdam/sequential_2/sequential_1/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOp8Adam/sequential_2/sequential_1/conv2d_transpose_1/bias/m*
_output_shapes
:@*
dtype0
Ў
:Adam/sequential_2/sequential_1/conv2d_transpose_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*K
shared_name<:Adam/sequential_2/sequential_1/conv2d_transpose_2/kernel/m
—
NAdam/sequential_2/sequential_1/conv2d_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOp:Adam/sequential_2/sequential_1/conv2d_transpose_2/kernel/m*&
_output_shapes
:@@*
dtype0
»
8Adam/sequential_2/sequential_1/conv2d_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adam/sequential_2/sequential_1/conv2d_transpose_2/bias/m
Ѕ
LAdam/sequential_2/sequential_1/conv2d_transpose_2/bias/m/Read/ReadVariableOpReadVariableOp8Adam/sequential_2/sequential_1/conv2d_transpose_2/bias/m*
_output_shapes
:@*
dtype0
Ў
:Adam/sequential_2/sequential_1/conv2d_transpose_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*K
shared_name<:Adam/sequential_2/sequential_1/conv2d_transpose_3/kernel/m
—
NAdam/sequential_2/sequential_1/conv2d_transpose_3/kernel/m/Read/ReadVariableOpReadVariableOp:Adam/sequential_2/sequential_1/conv2d_transpose_3/kernel/m*&
_output_shapes
:@*
dtype0
»
8Adam/sequential_2/sequential_1/conv2d_transpose_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/sequential_2/sequential_1/conv2d_transpose_3/bias/m
Ѕ
LAdam/sequential_2/sequential_1/conv2d_transpose_3/bias/m/Read/ReadVariableOpReadVariableOp8Adam/sequential_2/sequential_1/conv2d_transpose_3/bias/m*
_output_shapes
:*
dtype0
Љ
,Adam/sequential_2/sequential/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,Adam/sequential_2/sequential/conv2d/kernel/v
µ
@Adam/sequential_2/sequential/conv2d/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/sequential_2/sequential/conv2d/kernel/v*&
_output_shapes
:@*
dtype0
ђ
*Adam/sequential_2/sequential/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/sequential_2/sequential/conv2d/bias/v
•
>Adam/sequential_2/sequential/conv2d/bias/v/Read/ReadVariableOpReadVariableOp*Adam/sequential_2/sequential/conv2d/bias/v*
_output_shapes
:@*
dtype0
ј
.Adam/sequential_2/sequential/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*?
shared_name0.Adam/sequential_2/sequential/conv2d_1/kernel/v
є
BAdam/sequential_2/sequential/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/sequential_2/sequential/conv2d_1/kernel/v*&
_output_shapes
:@@*
dtype0
∞
,Adam/sequential_2/sequential/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,Adam/sequential_2/sequential/conv2d_1/bias/v
©
@Adam/sequential_2/sequential/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOp,Adam/sequential_2/sequential/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
ј
.Adam/sequential_2/sequential/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*?
shared_name0.Adam/sequential_2/sequential/conv2d_2/kernel/v
є
BAdam/sequential_2/sequential/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/sequential_2/sequential/conv2d_2/kernel/v*&
_output_shapes
:@@*
dtype0
∞
,Adam/sequential_2/sequential/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,Adam/sequential_2/sequential/conv2d_2/bias/v
©
@Adam/sequential_2/sequential/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOp,Adam/sequential_2/sequential/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
Ѕ
.Adam/sequential_2/sequential/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*?
shared_name0.Adam/sequential_2/sequential/conv2d_3/kernel/v
Ї
BAdam/sequential_2/sequential/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/sequential_2/sequential/conv2d_3/kernel/v*'
_output_shapes
:@А*
dtype0
±
,Adam/sequential_2/sequential/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*=
shared_name.,Adam/sequential_2/sequential/conv2d_3/bias/v
™
@Adam/sequential_2/sequential/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOp,Adam/sequential_2/sequential/conv2d_3/bias/v*
_output_shapes	
:А*
dtype0
≥
+Adam/sequential_2/sequential/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*<
shared_name-+Adam/sequential_2/sequential/dense/kernel/v
ђ
?Adam/sequential_2/sequential/dense/kernel/v/Read/ReadVariableOpReadVariableOp+Adam/sequential_2/sequential/dense/kernel/v*
_output_shapes
:	А*
dtype0
™
)Adam/sequential_2/sequential/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)Adam/sequential_2/sequential/dense/bias/v
£
=Adam/sequential_2/sequential/dense/bias/v/Read/ReadVariableOpReadVariableOp)Adam/sequential_2/sequential/dense/bias/v*
_output_shapes
:*
dtype0
ї
/Adam/sequential_2/sequential_1/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*@
shared_name1/Adam/sequential_2/sequential_1/dense_1/kernel/v
і
CAdam/sequential_2/sequential_1/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp/Adam/sequential_2/sequential_1/dense_1/kernel/v*
_output_shapes
:	А*
dtype0
≥
-Adam/sequential_2/sequential_1/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*>
shared_name/-Adam/sequential_2/sequential_1/dense_1/bias/v
ђ
AAdam/sequential_2/sequential_1/dense_1/bias/v/Read/ReadVariableOpReadVariableOp-Adam/sequential_2/sequential_1/dense_1/bias/v*
_output_shapes	
:А*
dtype0
’
8Adam/sequential_2/sequential_1/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*I
shared_name:8Adam/sequential_2/sequential_1/conv2d_transpose/kernel/v
ќ
LAdam/sequential_2/sequential_1/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOp8Adam/sequential_2/sequential_1/conv2d_transpose/kernel/v*'
_output_shapes
:@А*
dtype0
ƒ
6Adam/sequential_2/sequential_1/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*G
shared_name86Adam/sequential_2/sequential_1/conv2d_transpose/bias/v
љ
JAdam/sequential_2/sequential_1/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOp6Adam/sequential_2/sequential_1/conv2d_transpose/bias/v*
_output_shapes
:@*
dtype0
Ў
:Adam/sequential_2/sequential_1/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*K
shared_name<:Adam/sequential_2/sequential_1/conv2d_transpose_1/kernel/v
—
NAdam/sequential_2/sequential_1/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp:Adam/sequential_2/sequential_1/conv2d_transpose_1/kernel/v*&
_output_shapes
:@@*
dtype0
»
8Adam/sequential_2/sequential_1/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adam/sequential_2/sequential_1/conv2d_transpose_1/bias/v
Ѕ
LAdam/sequential_2/sequential_1/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOp8Adam/sequential_2/sequential_1/conv2d_transpose_1/bias/v*
_output_shapes
:@*
dtype0
Ў
:Adam/sequential_2/sequential_1/conv2d_transpose_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*K
shared_name<:Adam/sequential_2/sequential_1/conv2d_transpose_2/kernel/v
—
NAdam/sequential_2/sequential_1/conv2d_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOp:Adam/sequential_2/sequential_1/conv2d_transpose_2/kernel/v*&
_output_shapes
:@@*
dtype0
»
8Adam/sequential_2/sequential_1/conv2d_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adam/sequential_2/sequential_1/conv2d_transpose_2/bias/v
Ѕ
LAdam/sequential_2/sequential_1/conv2d_transpose_2/bias/v/Read/ReadVariableOpReadVariableOp8Adam/sequential_2/sequential_1/conv2d_transpose_2/bias/v*
_output_shapes
:@*
dtype0
Ў
:Adam/sequential_2/sequential_1/conv2d_transpose_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*K
shared_name<:Adam/sequential_2/sequential_1/conv2d_transpose_3/kernel/v
—
NAdam/sequential_2/sequential_1/conv2d_transpose_3/kernel/v/Read/ReadVariableOpReadVariableOp:Adam/sequential_2/sequential_1/conv2d_transpose_3/kernel/v*&
_output_shapes
:@*
dtype0
»
8Adam/sequential_2/sequential_1/conv2d_transpose_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*I
shared_name:8Adam/sequential_2/sequential_1/conv2d_transpose_3/bias/v
Ѕ
LAdam/sequential_2/sequential_1/conv2d_transpose_3/bias/v/Read/ReadVariableOpReadVariableOp8Adam/sequential_2/sequential_1/conv2d_transpose_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
іq
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*пp
valueеpBвp Bџp
Л
layer-0
layer-1
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
У
	layer-0

layer-1
layer-2
layer-3
layer-4
regularization_losses
trainable_variables
	variables
	keras_api
У
layer-0
layer-1
layer-2
layer-3
layer-4
regularization_losses
trainable_variables
	variables
	keras_api
–
iter

beta_1

beta_2
	decay
learning_rate mЬ!mЭ"mЮ#mЯ$m†%m°&mҐ'm£(m§)m•*m¶+mІ,m®-m©.m™/mЂ0mђ1m≠2mЃ3mѓ v∞!v±"v≤#v≥$vі%vµ&vґ'vЈ(vЄ)vє*vЇ+vї,vЉ-vљ.vЊ/vњ0vј1vЅ2v¬3v√
 
Ц
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14
/15
016
117
218
319
Ц
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14
/15
016
117
218
319
Ъ
4metrics

5layers
regularization_losses
trainable_variables
6non_trainable_variables
7layer_regularization_losses
	variables
 
h

 kernel
!bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
h

"kernel
#bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
h

$kernel
%bias
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
h

&kernel
'bias
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
h

(kernel
)bias
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
 
F
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
F
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
Ъ
Lmetrics

Mlayers
regularization_losses
trainable_variables
Nnon_trainable_variables
Olayer_regularization_losses
	variables
h

*kernel
+bias
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
h

,kernel
-bias
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
h

.kernel
/bias
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
h

0kernel
1bias
\regularization_losses
]trainable_variables
^	variables
_	keras_api
h

2kernel
3bias
`regularization_losses
atrainable_variables
b	variables
c	keras_api
 
F
*0
+1
,2
-3
.4
/5
06
17
28
39
F
*0
+1
,2
-3
.4
/5
06
17
28
39
Ъ
dmetrics

elayers
regularization_losses
trainable_variables
fnon_trainable_variables
glayer_regularization_losses
	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%sequential_2/sequential/conv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#sequential_2/sequential/conv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'sequential_2/sequential/conv2d_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%sequential_2/sequential/conv2d_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'sequential_2/sequential/conv2d_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%sequential_2/sequential/conv2d_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'sequential_2/sequential/conv2d_3/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%sequential_2/sequential/conv2d_3/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE$sequential_2/sequential/dense/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE"sequential_2/sequential/dense/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE(sequential_2/sequential_1/dense_1/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE&sequential_2/sequential_1/dense_1/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE1sequential_2/sequential_1/conv2d_transpose/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE/sequential_2/sequential_1/conv2d_transpose/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE3sequential_2/sequential_1/conv2d_transpose_1/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE1sequential_2/sequential_1/conv2d_transpose_1/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE3sequential_2/sequential_1/conv2d_transpose_2/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE1sequential_2/sequential_1/conv2d_transpose_2/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE3sequential_2/sequential_1/conv2d_transpose_3/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE1sequential_2/sequential_1/conv2d_transpose_3/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE

h0

0
1
 
 
 

 0
!1

 0
!1
Ъ
imetrics

jlayers
8regularization_losses
9trainable_variables
knon_trainable_variables
llayer_regularization_losses
:	variables
 

"0
#1

"0
#1
Ъ
mmetrics

nlayers
<regularization_losses
=trainable_variables
onon_trainable_variables
player_regularization_losses
>	variables
 

$0
%1

$0
%1
Ъ
qmetrics

rlayers
@regularization_losses
Atrainable_variables
snon_trainable_variables
tlayer_regularization_losses
B	variables
 

&0
'1

&0
'1
Ъ
umetrics

vlayers
Dregularization_losses
Etrainable_variables
wnon_trainable_variables
xlayer_regularization_losses
F	variables
 

(0
)1

(0
)1
Ъ
ymetrics

zlayers
Hregularization_losses
Itrainable_variables
{non_trainable_variables
|layer_regularization_losses
J	variables
 
#
	0

1
2
3
4
 
 
 

*0
+1

*0
+1
Ы
}metrics

~layers
Pregularization_losses
Qtrainable_variables
non_trainable_variables
 Аlayer_regularization_losses
R	variables
 

,0
-1

,0
-1
Ю
Бmetrics
Вlayers
Tregularization_losses
Utrainable_variables
Гnon_trainable_variables
 Дlayer_regularization_losses
V	variables
 

.0
/1

.0
/1
Ю
Еmetrics
Жlayers
Xregularization_losses
Ytrainable_variables
Зnon_trainable_variables
 Иlayer_regularization_losses
Z	variables
 

00
11

00
11
Ю
Йmetrics
Кlayers
\regularization_losses
]trainable_variables
Лnon_trainable_variables
 Мlayer_regularization_losses
^	variables
 

20
31

20
31
Ю
Нmetrics
Оlayers
`regularization_losses
atrainable_variables
Пnon_trainable_variables
 Рlayer_regularization_losses
b	variables
 
#
0
1
2
3
4
 
 


Сtotal

Тcount
У
_fn_kwargs
Фregularization_losses
Хtrainable_variables
Ц	variables
Ч	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

С0
Т1
°
Шmetrics
Щlayers
Фregularization_losses
Хtrainable_variables
Ъnon_trainable_variables
 Ыlayer_regularization_losses
Ц	variables
 
 

С0
Т1
 
ПМ
VARIABLE_VALUE,Adam/sequential_2/sequential/conv2d/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*Adam/sequential_2/sequential/conv2d/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE.Adam/sequential_2/sequential/conv2d_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/sequential_2/sequential/conv2d_1/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE.Adam/sequential_2/sequential/conv2d_2/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/sequential_2/sequential/conv2d_2/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE.Adam/sequential_2/sequential/conv2d_3/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/sequential_2/sequential/conv2d_3/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ОЛ
VARIABLE_VALUE+Adam/sequential_2/sequential/dense/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE)Adam/sequential_2/sequential/dense/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE/Adam/sequential_2/sequential_1/dense_1/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE-Adam/sequential_2/sequential_1/dense_1/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUE8Adam/sequential_2/sequential_1/conv2d_transpose/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE6Adam/sequential_2/sequential_1/conv2d_transpose/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE:Adam/sequential_2/sequential_1/conv2d_transpose_1/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUE8Adam/sequential_2/sequential_1/conv2d_transpose_1/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE:Adam/sequential_2/sequential_1/conv2d_transpose_2/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUE8Adam/sequential_2/sequential_1/conv2d_transpose_2/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE:Adam/sequential_2/sequential_1/conv2d_transpose_3/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUE8Adam/sequential_2/sequential_1/conv2d_transpose_3/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/sequential_2/sequential/conv2d/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*Adam/sequential_2/sequential/conv2d/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE.Adam/sequential_2/sequential/conv2d_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/sequential_2/sequential/conv2d_1/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE.Adam/sequential_2/sequential/conv2d_2/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/sequential_2/sequential/conv2d_2/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE.Adam/sequential_2/sequential/conv2d_3/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/sequential_2/sequential/conv2d_3/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ОЛ
VARIABLE_VALUE+Adam/sequential_2/sequential/dense/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE)Adam/sequential_2/sequential/dense/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
УР
VARIABLE_VALUE/Adam/sequential_2/sequential_1/dense_1/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
СО
VARIABLE_VALUE-Adam/sequential_2/sequential_1/dense_1/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUE8Adam/sequential_2/sequential_1/conv2d_transpose/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE6Adam/sequential_2/sequential_1/conv2d_transpose/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE:Adam/sequential_2/sequential_1/conv2d_transpose_1/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUE8Adam/sequential_2/sequential_1/conv2d_transpose_1/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE:Adam/sequential_2/sequential_1/conv2d_transpose_2/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUE8Adam/sequential_2/sequential_1/conv2d_transpose_2/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЮЫ
VARIABLE_VALUE:Adam/sequential_2/sequential_1/conv2d_transpose_3/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЬЩ
VARIABLE_VALUE8Adam/sequential_2/sequential_1/conv2d_transpose_3/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
М
serving_default_input_1Placeholder*0
_output_shapes
:€€€€€€€€€`ј*
dtype0*%
shape:€€€€€€€€€`ј
™	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1%sequential_2/sequential/conv2d/kernel#sequential_2/sequential/conv2d/bias'sequential_2/sequential/conv2d_1/kernel%sequential_2/sequential/conv2d_1/bias'sequential_2/sequential/conv2d_2/kernel%sequential_2/sequential/conv2d_2/bias'sequential_2/sequential/conv2d_3/kernel%sequential_2/sequential/conv2d_3/bias$sequential_2/sequential/dense/kernel"sequential_2/sequential/dense/bias(sequential_2/sequential_1/dense_1/kernel&sequential_2/sequential_1/dense_1/bias1sequential_2/sequential_1/conv2d_transpose/kernel/sequential_2/sequential_1/conv2d_transpose/bias3sequential_2/sequential_1/conv2d_transpose_1/kernel1sequential_2/sequential_1/conv2d_transpose_1/bias3sequential_2/sequential_1/conv2d_transpose_2/kernel1sequential_2/sequential_1/conv2d_transpose_2/bias3sequential_2/sequential_1/conv2d_transpose_3/kernel1sequential_2/sequential_1/conv2d_transpose_3/bias* 
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€`ј**
config_proto

GPU 

CPU2J 8*,
f'R%
#__inference_signature_wrapper_36989
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
 $
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9sequential_2/sequential/conv2d/kernel/Read/ReadVariableOp7sequential_2/sequential/conv2d/bias/Read/ReadVariableOp;sequential_2/sequential/conv2d_1/kernel/Read/ReadVariableOp9sequential_2/sequential/conv2d_1/bias/Read/ReadVariableOp;sequential_2/sequential/conv2d_2/kernel/Read/ReadVariableOp9sequential_2/sequential/conv2d_2/bias/Read/ReadVariableOp;sequential_2/sequential/conv2d_3/kernel/Read/ReadVariableOp9sequential_2/sequential/conv2d_3/bias/Read/ReadVariableOp8sequential_2/sequential/dense/kernel/Read/ReadVariableOp6sequential_2/sequential/dense/bias/Read/ReadVariableOp<sequential_2/sequential_1/dense_1/kernel/Read/ReadVariableOp:sequential_2/sequential_1/dense_1/bias/Read/ReadVariableOpEsequential_2/sequential_1/conv2d_transpose/kernel/Read/ReadVariableOpCsequential_2/sequential_1/conv2d_transpose/bias/Read/ReadVariableOpGsequential_2/sequential_1/conv2d_transpose_1/kernel/Read/ReadVariableOpEsequential_2/sequential_1/conv2d_transpose_1/bias/Read/ReadVariableOpGsequential_2/sequential_1/conv2d_transpose_2/kernel/Read/ReadVariableOpEsequential_2/sequential_1/conv2d_transpose_2/bias/Read/ReadVariableOpGsequential_2/sequential_1/conv2d_transpose_3/kernel/Read/ReadVariableOpEsequential_2/sequential_1/conv2d_transpose_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp@Adam/sequential_2/sequential/conv2d/kernel/m/Read/ReadVariableOp>Adam/sequential_2/sequential/conv2d/bias/m/Read/ReadVariableOpBAdam/sequential_2/sequential/conv2d_1/kernel/m/Read/ReadVariableOp@Adam/sequential_2/sequential/conv2d_1/bias/m/Read/ReadVariableOpBAdam/sequential_2/sequential/conv2d_2/kernel/m/Read/ReadVariableOp@Adam/sequential_2/sequential/conv2d_2/bias/m/Read/ReadVariableOpBAdam/sequential_2/sequential/conv2d_3/kernel/m/Read/ReadVariableOp@Adam/sequential_2/sequential/conv2d_3/bias/m/Read/ReadVariableOp?Adam/sequential_2/sequential/dense/kernel/m/Read/ReadVariableOp=Adam/sequential_2/sequential/dense/bias/m/Read/ReadVariableOpCAdam/sequential_2/sequential_1/dense_1/kernel/m/Read/ReadVariableOpAAdam/sequential_2/sequential_1/dense_1/bias/m/Read/ReadVariableOpLAdam/sequential_2/sequential_1/conv2d_transpose/kernel/m/Read/ReadVariableOpJAdam/sequential_2/sequential_1/conv2d_transpose/bias/m/Read/ReadVariableOpNAdam/sequential_2/sequential_1/conv2d_transpose_1/kernel/m/Read/ReadVariableOpLAdam/sequential_2/sequential_1/conv2d_transpose_1/bias/m/Read/ReadVariableOpNAdam/sequential_2/sequential_1/conv2d_transpose_2/kernel/m/Read/ReadVariableOpLAdam/sequential_2/sequential_1/conv2d_transpose_2/bias/m/Read/ReadVariableOpNAdam/sequential_2/sequential_1/conv2d_transpose_3/kernel/m/Read/ReadVariableOpLAdam/sequential_2/sequential_1/conv2d_transpose_3/bias/m/Read/ReadVariableOp@Adam/sequential_2/sequential/conv2d/kernel/v/Read/ReadVariableOp>Adam/sequential_2/sequential/conv2d/bias/v/Read/ReadVariableOpBAdam/sequential_2/sequential/conv2d_1/kernel/v/Read/ReadVariableOp@Adam/sequential_2/sequential/conv2d_1/bias/v/Read/ReadVariableOpBAdam/sequential_2/sequential/conv2d_2/kernel/v/Read/ReadVariableOp@Adam/sequential_2/sequential/conv2d_2/bias/v/Read/ReadVariableOpBAdam/sequential_2/sequential/conv2d_3/kernel/v/Read/ReadVariableOp@Adam/sequential_2/sequential/conv2d_3/bias/v/Read/ReadVariableOp?Adam/sequential_2/sequential/dense/kernel/v/Read/ReadVariableOp=Adam/sequential_2/sequential/dense/bias/v/Read/ReadVariableOpCAdam/sequential_2/sequential_1/dense_1/kernel/v/Read/ReadVariableOpAAdam/sequential_2/sequential_1/dense_1/bias/v/Read/ReadVariableOpLAdam/sequential_2/sequential_1/conv2d_transpose/kernel/v/Read/ReadVariableOpJAdam/sequential_2/sequential_1/conv2d_transpose/bias/v/Read/ReadVariableOpNAdam/sequential_2/sequential_1/conv2d_transpose_1/kernel/v/Read/ReadVariableOpLAdam/sequential_2/sequential_1/conv2d_transpose_1/bias/v/Read/ReadVariableOpNAdam/sequential_2/sequential_1/conv2d_transpose_2/kernel/v/Read/ReadVariableOpLAdam/sequential_2/sequential_1/conv2d_transpose_2/bias/v/Read/ReadVariableOpNAdam/sequential_2/sequential_1/conv2d_transpose_3/kernel/v/Read/ReadVariableOpLAdam/sequential_2/sequential_1/conv2d_transpose_3/bias/v/Read/ReadVariableOpConst*P
TinI
G2E	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

GPU 

CPU2J 8*'
f"R 
__inference__traced_save_38326
Й
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%sequential_2/sequential/conv2d/kernel#sequential_2/sequential/conv2d/bias'sequential_2/sequential/conv2d_1/kernel%sequential_2/sequential/conv2d_1/bias'sequential_2/sequential/conv2d_2/kernel%sequential_2/sequential/conv2d_2/bias'sequential_2/sequential/conv2d_3/kernel%sequential_2/sequential/conv2d_3/bias$sequential_2/sequential/dense/kernel"sequential_2/sequential/dense/bias(sequential_2/sequential_1/dense_1/kernel&sequential_2/sequential_1/dense_1/bias1sequential_2/sequential_1/conv2d_transpose/kernel/sequential_2/sequential_1/conv2d_transpose/bias3sequential_2/sequential_1/conv2d_transpose_1/kernel1sequential_2/sequential_1/conv2d_transpose_1/bias3sequential_2/sequential_1/conv2d_transpose_2/kernel1sequential_2/sequential_1/conv2d_transpose_2/bias3sequential_2/sequential_1/conv2d_transpose_3/kernel1sequential_2/sequential_1/conv2d_transpose_3/biastotalcount,Adam/sequential_2/sequential/conv2d/kernel/m*Adam/sequential_2/sequential/conv2d/bias/m.Adam/sequential_2/sequential/conv2d_1/kernel/m,Adam/sequential_2/sequential/conv2d_1/bias/m.Adam/sequential_2/sequential/conv2d_2/kernel/m,Adam/sequential_2/sequential/conv2d_2/bias/m.Adam/sequential_2/sequential/conv2d_3/kernel/m,Adam/sequential_2/sequential/conv2d_3/bias/m+Adam/sequential_2/sequential/dense/kernel/m)Adam/sequential_2/sequential/dense/bias/m/Adam/sequential_2/sequential_1/dense_1/kernel/m-Adam/sequential_2/sequential_1/dense_1/bias/m8Adam/sequential_2/sequential_1/conv2d_transpose/kernel/m6Adam/sequential_2/sequential_1/conv2d_transpose/bias/m:Adam/sequential_2/sequential_1/conv2d_transpose_1/kernel/m8Adam/sequential_2/sequential_1/conv2d_transpose_1/bias/m:Adam/sequential_2/sequential_1/conv2d_transpose_2/kernel/m8Adam/sequential_2/sequential_1/conv2d_transpose_2/bias/m:Adam/sequential_2/sequential_1/conv2d_transpose_3/kernel/m8Adam/sequential_2/sequential_1/conv2d_transpose_3/bias/m,Adam/sequential_2/sequential/conv2d/kernel/v*Adam/sequential_2/sequential/conv2d/bias/v.Adam/sequential_2/sequential/conv2d_1/kernel/v,Adam/sequential_2/sequential/conv2d_1/bias/v.Adam/sequential_2/sequential/conv2d_2/kernel/v,Adam/sequential_2/sequential/conv2d_2/bias/v.Adam/sequential_2/sequential/conv2d_3/kernel/v,Adam/sequential_2/sequential/conv2d_3/bias/v+Adam/sequential_2/sequential/dense/kernel/v)Adam/sequential_2/sequential/dense/bias/v/Adam/sequential_2/sequential_1/dense_1/kernel/v-Adam/sequential_2/sequential_1/dense_1/bias/v8Adam/sequential_2/sequential_1/conv2d_transpose/kernel/v6Adam/sequential_2/sequential_1/conv2d_transpose/bias/v:Adam/sequential_2/sequential_1/conv2d_transpose_1/kernel/v8Adam/sequential_2/sequential_1/conv2d_transpose_1/bias/v:Adam/sequential_2/sequential_1/conv2d_transpose_2/kernel/v8Adam/sequential_2/sequential_1/conv2d_transpose_2/bias/v:Adam/sequential_2/sequential_1/conv2d_transpose_3/kernel/v8Adam/sequential_2/sequential_1/conv2d_transpose_3/bias/v*O
TinH
F2D*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: **
config_proto

GPU 

CPU2J 8**
f%R#
!__inference__traced_restore_38539√О
ћО
е(
__inference__traced_save_38326
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_sequential_2_sequential_conv2d_kernel_read_readvariableopB
>savev2_sequential_2_sequential_conv2d_bias_read_readvariableopF
Bsavev2_sequential_2_sequential_conv2d_1_kernel_read_readvariableopD
@savev2_sequential_2_sequential_conv2d_1_bias_read_readvariableopF
Bsavev2_sequential_2_sequential_conv2d_2_kernel_read_readvariableopD
@savev2_sequential_2_sequential_conv2d_2_bias_read_readvariableopF
Bsavev2_sequential_2_sequential_conv2d_3_kernel_read_readvariableopD
@savev2_sequential_2_sequential_conv2d_3_bias_read_readvariableopC
?savev2_sequential_2_sequential_dense_kernel_read_readvariableopA
=savev2_sequential_2_sequential_dense_bias_read_readvariableopG
Csavev2_sequential_2_sequential_1_dense_1_kernel_read_readvariableopE
Asavev2_sequential_2_sequential_1_dense_1_bias_read_readvariableopP
Lsavev2_sequential_2_sequential_1_conv2d_transpose_kernel_read_readvariableopN
Jsavev2_sequential_2_sequential_1_conv2d_transpose_bias_read_readvariableopR
Nsavev2_sequential_2_sequential_1_conv2d_transpose_1_kernel_read_readvariableopP
Lsavev2_sequential_2_sequential_1_conv2d_transpose_1_bias_read_readvariableopR
Nsavev2_sequential_2_sequential_1_conv2d_transpose_2_kernel_read_readvariableopP
Lsavev2_sequential_2_sequential_1_conv2d_transpose_2_bias_read_readvariableopR
Nsavev2_sequential_2_sequential_1_conv2d_transpose_3_kernel_read_readvariableopP
Lsavev2_sequential_2_sequential_1_conv2d_transpose_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopK
Gsavev2_adam_sequential_2_sequential_conv2d_kernel_m_read_readvariableopI
Esavev2_adam_sequential_2_sequential_conv2d_bias_m_read_readvariableopM
Isavev2_adam_sequential_2_sequential_conv2d_1_kernel_m_read_readvariableopK
Gsavev2_adam_sequential_2_sequential_conv2d_1_bias_m_read_readvariableopM
Isavev2_adam_sequential_2_sequential_conv2d_2_kernel_m_read_readvariableopK
Gsavev2_adam_sequential_2_sequential_conv2d_2_bias_m_read_readvariableopM
Isavev2_adam_sequential_2_sequential_conv2d_3_kernel_m_read_readvariableopK
Gsavev2_adam_sequential_2_sequential_conv2d_3_bias_m_read_readvariableopJ
Fsavev2_adam_sequential_2_sequential_dense_kernel_m_read_readvariableopH
Dsavev2_adam_sequential_2_sequential_dense_bias_m_read_readvariableopN
Jsavev2_adam_sequential_2_sequential_1_dense_1_kernel_m_read_readvariableopL
Hsavev2_adam_sequential_2_sequential_1_dense_1_bias_m_read_readvariableopW
Ssavev2_adam_sequential_2_sequential_1_conv2d_transpose_kernel_m_read_readvariableopU
Qsavev2_adam_sequential_2_sequential_1_conv2d_transpose_bias_m_read_readvariableopY
Usavev2_adam_sequential_2_sequential_1_conv2d_transpose_1_kernel_m_read_readvariableopW
Ssavev2_adam_sequential_2_sequential_1_conv2d_transpose_1_bias_m_read_readvariableopY
Usavev2_adam_sequential_2_sequential_1_conv2d_transpose_2_kernel_m_read_readvariableopW
Ssavev2_adam_sequential_2_sequential_1_conv2d_transpose_2_bias_m_read_readvariableopY
Usavev2_adam_sequential_2_sequential_1_conv2d_transpose_3_kernel_m_read_readvariableopW
Ssavev2_adam_sequential_2_sequential_1_conv2d_transpose_3_bias_m_read_readvariableopK
Gsavev2_adam_sequential_2_sequential_conv2d_kernel_v_read_readvariableopI
Esavev2_adam_sequential_2_sequential_conv2d_bias_v_read_readvariableopM
Isavev2_adam_sequential_2_sequential_conv2d_1_kernel_v_read_readvariableopK
Gsavev2_adam_sequential_2_sequential_conv2d_1_bias_v_read_readvariableopM
Isavev2_adam_sequential_2_sequential_conv2d_2_kernel_v_read_readvariableopK
Gsavev2_adam_sequential_2_sequential_conv2d_2_bias_v_read_readvariableopM
Isavev2_adam_sequential_2_sequential_conv2d_3_kernel_v_read_readvariableopK
Gsavev2_adam_sequential_2_sequential_conv2d_3_bias_v_read_readvariableopJ
Fsavev2_adam_sequential_2_sequential_dense_kernel_v_read_readvariableopH
Dsavev2_adam_sequential_2_sequential_dense_bias_v_read_readvariableopN
Jsavev2_adam_sequential_2_sequential_1_dense_1_kernel_v_read_readvariableopL
Hsavev2_adam_sequential_2_sequential_1_dense_1_bias_v_read_readvariableopW
Ssavev2_adam_sequential_2_sequential_1_conv2d_transpose_kernel_v_read_readvariableopU
Qsavev2_adam_sequential_2_sequential_1_conv2d_transpose_bias_v_read_readvariableopY
Usavev2_adam_sequential_2_sequential_1_conv2d_transpose_1_kernel_v_read_readvariableopW
Ssavev2_adam_sequential_2_sequential_1_conv2d_transpose_1_bias_v_read_readvariableopY
Usavev2_adam_sequential_2_sequential_1_conv2d_transpose_2_kernel_v_read_readvariableopW
Ssavev2_adam_sequential_2_sequential_1_conv2d_transpose_2_bias_v_read_readvariableopY
Usavev2_adam_sequential_2_sequential_1_conv2d_transpose_3_kernel_v_read_readvariableopW
Ssavev2_adam_sequential_2_sequential_1_conv2d_transpose_3_bias_v_read_readvariableop
savev2_1_const

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1•
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_c5fa5ad7a03348cab4ebadd1a15567c1/part2
StringJoin/inputs_1Б

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameК$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*Ь#
valueТ#BП#CB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesС
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*Ы
valueСBОCB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices∆'
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_sequential_2_sequential_conv2d_kernel_read_readvariableop>savev2_sequential_2_sequential_conv2d_bias_read_readvariableopBsavev2_sequential_2_sequential_conv2d_1_kernel_read_readvariableop@savev2_sequential_2_sequential_conv2d_1_bias_read_readvariableopBsavev2_sequential_2_sequential_conv2d_2_kernel_read_readvariableop@savev2_sequential_2_sequential_conv2d_2_bias_read_readvariableopBsavev2_sequential_2_sequential_conv2d_3_kernel_read_readvariableop@savev2_sequential_2_sequential_conv2d_3_bias_read_readvariableop?savev2_sequential_2_sequential_dense_kernel_read_readvariableop=savev2_sequential_2_sequential_dense_bias_read_readvariableopCsavev2_sequential_2_sequential_1_dense_1_kernel_read_readvariableopAsavev2_sequential_2_sequential_1_dense_1_bias_read_readvariableopLsavev2_sequential_2_sequential_1_conv2d_transpose_kernel_read_readvariableopJsavev2_sequential_2_sequential_1_conv2d_transpose_bias_read_readvariableopNsavev2_sequential_2_sequential_1_conv2d_transpose_1_kernel_read_readvariableopLsavev2_sequential_2_sequential_1_conv2d_transpose_1_bias_read_readvariableopNsavev2_sequential_2_sequential_1_conv2d_transpose_2_kernel_read_readvariableopLsavev2_sequential_2_sequential_1_conv2d_transpose_2_bias_read_readvariableopNsavev2_sequential_2_sequential_1_conv2d_transpose_3_kernel_read_readvariableopLsavev2_sequential_2_sequential_1_conv2d_transpose_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopGsavev2_adam_sequential_2_sequential_conv2d_kernel_m_read_readvariableopEsavev2_adam_sequential_2_sequential_conv2d_bias_m_read_readvariableopIsavev2_adam_sequential_2_sequential_conv2d_1_kernel_m_read_readvariableopGsavev2_adam_sequential_2_sequential_conv2d_1_bias_m_read_readvariableopIsavev2_adam_sequential_2_sequential_conv2d_2_kernel_m_read_readvariableopGsavev2_adam_sequential_2_sequential_conv2d_2_bias_m_read_readvariableopIsavev2_adam_sequential_2_sequential_conv2d_3_kernel_m_read_readvariableopGsavev2_adam_sequential_2_sequential_conv2d_3_bias_m_read_readvariableopFsavev2_adam_sequential_2_sequential_dense_kernel_m_read_readvariableopDsavev2_adam_sequential_2_sequential_dense_bias_m_read_readvariableopJsavev2_adam_sequential_2_sequential_1_dense_1_kernel_m_read_readvariableopHsavev2_adam_sequential_2_sequential_1_dense_1_bias_m_read_readvariableopSsavev2_adam_sequential_2_sequential_1_conv2d_transpose_kernel_m_read_readvariableopQsavev2_adam_sequential_2_sequential_1_conv2d_transpose_bias_m_read_readvariableopUsavev2_adam_sequential_2_sequential_1_conv2d_transpose_1_kernel_m_read_readvariableopSsavev2_adam_sequential_2_sequential_1_conv2d_transpose_1_bias_m_read_readvariableopUsavev2_adam_sequential_2_sequential_1_conv2d_transpose_2_kernel_m_read_readvariableopSsavev2_adam_sequential_2_sequential_1_conv2d_transpose_2_bias_m_read_readvariableopUsavev2_adam_sequential_2_sequential_1_conv2d_transpose_3_kernel_m_read_readvariableopSsavev2_adam_sequential_2_sequential_1_conv2d_transpose_3_bias_m_read_readvariableopGsavev2_adam_sequential_2_sequential_conv2d_kernel_v_read_readvariableopEsavev2_adam_sequential_2_sequential_conv2d_bias_v_read_readvariableopIsavev2_adam_sequential_2_sequential_conv2d_1_kernel_v_read_readvariableopGsavev2_adam_sequential_2_sequential_conv2d_1_bias_v_read_readvariableopIsavev2_adam_sequential_2_sequential_conv2d_2_kernel_v_read_readvariableopGsavev2_adam_sequential_2_sequential_conv2d_2_bias_v_read_readvariableopIsavev2_adam_sequential_2_sequential_conv2d_3_kernel_v_read_readvariableopGsavev2_adam_sequential_2_sequential_conv2d_3_bias_v_read_readvariableopFsavev2_adam_sequential_2_sequential_dense_kernel_v_read_readvariableopDsavev2_adam_sequential_2_sequential_dense_bias_v_read_readvariableopJsavev2_adam_sequential_2_sequential_1_dense_1_kernel_v_read_readvariableopHsavev2_adam_sequential_2_sequential_1_dense_1_bias_v_read_readvariableopSsavev2_adam_sequential_2_sequential_1_conv2d_transpose_kernel_v_read_readvariableopQsavev2_adam_sequential_2_sequential_1_conv2d_transpose_bias_v_read_readvariableopUsavev2_adam_sequential_2_sequential_1_conv2d_transpose_1_kernel_v_read_readvariableopSsavev2_adam_sequential_2_sequential_1_conv2d_transpose_1_bias_v_read_readvariableopUsavev2_adam_sequential_2_sequential_1_conv2d_transpose_2_kernel_v_read_readvariableopSsavev2_adam_sequential_2_sequential_1_conv2d_transpose_2_bias_v_read_readvariableopUsavev2_adam_sequential_2_sequential_1_conv2d_transpose_3_kernel_v_read_readvariableopSsavev2_adam_sequential_2_sequential_1_conv2d_transpose_3_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *Q
dtypesG
E2C	2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardђ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ґ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesѕ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1г
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesђ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ў
_input_shapes«
ƒ: : : : : : :@:@:@@:@:@@:@:@А:А:	А::	А:А:@А:@:@@:@:@@:@:@:: : :@:@:@@:@:@@:@:@А:А:	А::	А:А:@А:@:@@:@:@@:@:@::@:@:@@:@:@@:@:@А:А:	А::	А:А:@А:@:@@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
£$
б
B__inference_dense_1_layer_call_and_return_conditional_losses_36626

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis—
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis„
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/ConstА
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1И
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackФ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЙ
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/transpose_1/permђ
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	А2
Tensordot/transpose_1З
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   А   2
Tensordot/Reshape_1/shapeЮ
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2
Tensordot/Reshape_1Ы
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Х
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
	TensordotН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdd°
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
И
®
'__inference_dense_1_layer_call_fn_38101

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€А**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_366262
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
н&
ъ
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_36533

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2м
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3В
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3≥
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_transpose/ReadVariableOpс
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
Reluї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
и
№
C__inference_conv2d_2_layer_call_and_return_conditional_losses_36220

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpґ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
ь•
Ы0
!__inference__traced_restore_38539
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate<
8assignvariableop_5_sequential_2_sequential_conv2d_kernel:
6assignvariableop_6_sequential_2_sequential_conv2d_bias>
:assignvariableop_7_sequential_2_sequential_conv2d_1_kernel<
8assignvariableop_8_sequential_2_sequential_conv2d_1_bias>
:assignvariableop_9_sequential_2_sequential_conv2d_2_kernel=
9assignvariableop_10_sequential_2_sequential_conv2d_2_bias?
;assignvariableop_11_sequential_2_sequential_conv2d_3_kernel=
9assignvariableop_12_sequential_2_sequential_conv2d_3_bias<
8assignvariableop_13_sequential_2_sequential_dense_kernel:
6assignvariableop_14_sequential_2_sequential_dense_bias@
<assignvariableop_15_sequential_2_sequential_1_dense_1_kernel>
:assignvariableop_16_sequential_2_sequential_1_dense_1_biasI
Eassignvariableop_17_sequential_2_sequential_1_conv2d_transpose_kernelG
Cassignvariableop_18_sequential_2_sequential_1_conv2d_transpose_biasK
Gassignvariableop_19_sequential_2_sequential_1_conv2d_transpose_1_kernelI
Eassignvariableop_20_sequential_2_sequential_1_conv2d_transpose_1_biasK
Gassignvariableop_21_sequential_2_sequential_1_conv2d_transpose_2_kernelI
Eassignvariableop_22_sequential_2_sequential_1_conv2d_transpose_2_biasK
Gassignvariableop_23_sequential_2_sequential_1_conv2d_transpose_3_kernelI
Eassignvariableop_24_sequential_2_sequential_1_conv2d_transpose_3_bias
assignvariableop_25_total
assignvariableop_26_countD
@assignvariableop_27_adam_sequential_2_sequential_conv2d_kernel_mB
>assignvariableop_28_adam_sequential_2_sequential_conv2d_bias_mF
Bassignvariableop_29_adam_sequential_2_sequential_conv2d_1_kernel_mD
@assignvariableop_30_adam_sequential_2_sequential_conv2d_1_bias_mF
Bassignvariableop_31_adam_sequential_2_sequential_conv2d_2_kernel_mD
@assignvariableop_32_adam_sequential_2_sequential_conv2d_2_bias_mF
Bassignvariableop_33_adam_sequential_2_sequential_conv2d_3_kernel_mD
@assignvariableop_34_adam_sequential_2_sequential_conv2d_3_bias_mC
?assignvariableop_35_adam_sequential_2_sequential_dense_kernel_mA
=assignvariableop_36_adam_sequential_2_sequential_dense_bias_mG
Cassignvariableop_37_adam_sequential_2_sequential_1_dense_1_kernel_mE
Aassignvariableop_38_adam_sequential_2_sequential_1_dense_1_bias_mP
Lassignvariableop_39_adam_sequential_2_sequential_1_conv2d_transpose_kernel_mN
Jassignvariableop_40_adam_sequential_2_sequential_1_conv2d_transpose_bias_mR
Nassignvariableop_41_adam_sequential_2_sequential_1_conv2d_transpose_1_kernel_mP
Lassignvariableop_42_adam_sequential_2_sequential_1_conv2d_transpose_1_bias_mR
Nassignvariableop_43_adam_sequential_2_sequential_1_conv2d_transpose_2_kernel_mP
Lassignvariableop_44_adam_sequential_2_sequential_1_conv2d_transpose_2_bias_mR
Nassignvariableop_45_adam_sequential_2_sequential_1_conv2d_transpose_3_kernel_mP
Lassignvariableop_46_adam_sequential_2_sequential_1_conv2d_transpose_3_bias_mD
@assignvariableop_47_adam_sequential_2_sequential_conv2d_kernel_vB
>assignvariableop_48_adam_sequential_2_sequential_conv2d_bias_vF
Bassignvariableop_49_adam_sequential_2_sequential_conv2d_1_kernel_vD
@assignvariableop_50_adam_sequential_2_sequential_conv2d_1_bias_vF
Bassignvariableop_51_adam_sequential_2_sequential_conv2d_2_kernel_vD
@assignvariableop_52_adam_sequential_2_sequential_conv2d_2_bias_vF
Bassignvariableop_53_adam_sequential_2_sequential_conv2d_3_kernel_vD
@assignvariableop_54_adam_sequential_2_sequential_conv2d_3_bias_vC
?assignvariableop_55_adam_sequential_2_sequential_dense_kernel_vA
=assignvariableop_56_adam_sequential_2_sequential_dense_bias_vG
Cassignvariableop_57_adam_sequential_2_sequential_1_dense_1_kernel_vE
Aassignvariableop_58_adam_sequential_2_sequential_1_dense_1_bias_vP
Lassignvariableop_59_adam_sequential_2_sequential_1_conv2d_transpose_kernel_vN
Jassignvariableop_60_adam_sequential_2_sequential_1_conv2d_transpose_bias_vR
Nassignvariableop_61_adam_sequential_2_sequential_1_conv2d_transpose_1_kernel_vP
Lassignvariableop_62_adam_sequential_2_sequential_1_conv2d_transpose_1_bias_vR
Nassignvariableop_63_adam_sequential_2_sequential_1_conv2d_transpose_2_kernel_vP
Lassignvariableop_64_adam_sequential_2_sequential_1_conv2d_transpose_2_bias_vR
Nassignvariableop_65_adam_sequential_2_sequential_1_conv2d_transpose_3_kernel_vP
Lassignvariableop_66_adam_sequential_2_sequential_1_conv2d_transpose_3_bias_v
identity_68ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ґ	RestoreV2ҐRestoreV2_1Р$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*Ь#
valueТ#BП#CB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesЧ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*Ы
valueСBОCB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesэ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ґ
_output_shapesП
М:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Q
dtypesG
E2C	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0	*
_output_shapes
:2

IdentityК
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ф
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ф
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3У
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ы
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Ѓ
AssignVariableOp_5AssignVariableOp8assignvariableop_5_sequential_2_sequential_conv2d_kernelIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6ђ
AssignVariableOp_6AssignVariableOp6assignvariableop_6_sequential_2_sequential_conv2d_biasIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7∞
AssignVariableOp_7AssignVariableOp:assignvariableop_7_sequential_2_sequential_conv2d_1_kernelIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Ѓ
AssignVariableOp_8AssignVariableOp8assignvariableop_8_sequential_2_sequential_conv2d_1_biasIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9∞
AssignVariableOp_9AssignVariableOp:assignvariableop_9_sequential_2_sequential_conv2d_2_kernelIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10≤
AssignVariableOp_10AssignVariableOp9assignvariableop_10_sequential_2_sequential_conv2d_2_biasIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11і
AssignVariableOp_11AssignVariableOp;assignvariableop_11_sequential_2_sequential_conv2d_3_kernelIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12≤
AssignVariableOp_12AssignVariableOp9assignvariableop_12_sequential_2_sequential_conv2d_3_biasIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13±
AssignVariableOp_13AssignVariableOp8assignvariableop_13_sequential_2_sequential_dense_kernelIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14ѓ
AssignVariableOp_14AssignVariableOp6assignvariableop_14_sequential_2_sequential_dense_biasIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15µ
AssignVariableOp_15AssignVariableOp<assignvariableop_15_sequential_2_sequential_1_dense_1_kernelIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16≥
AssignVariableOp_16AssignVariableOp:assignvariableop_16_sequential_2_sequential_1_dense_1_biasIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Њ
AssignVariableOp_17AssignVariableOpEassignvariableop_17_sequential_2_sequential_1_conv2d_transpose_kernelIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18Љ
AssignVariableOp_18AssignVariableOpCassignvariableop_18_sequential_2_sequential_1_conv2d_transpose_biasIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19ј
AssignVariableOp_19AssignVariableOpGassignvariableop_19_sequential_2_sequential_1_conv2d_transpose_1_kernelIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20Њ
AssignVariableOp_20AssignVariableOpEassignvariableop_20_sequential_2_sequential_1_conv2d_transpose_1_biasIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21ј
AssignVariableOp_21AssignVariableOpGassignvariableop_21_sequential_2_sequential_1_conv2d_transpose_2_kernelIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22Њ
AssignVariableOp_22AssignVariableOpEassignvariableop_22_sequential_2_sequential_1_conv2d_transpose_2_biasIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23ј
AssignVariableOp_23AssignVariableOpGassignvariableop_23_sequential_2_sequential_1_conv2d_transpose_3_kernelIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24Њ
AssignVariableOp_24AssignVariableOpEassignvariableop_24_sequential_2_sequential_1_conv2d_transpose_3_biasIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25Т
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26Т
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27є
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_sequential_2_sequential_conv2d_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28Ј
AssignVariableOp_28AssignVariableOp>assignvariableop_28_adam_sequential_2_sequential_conv2d_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29ї
AssignVariableOp_29AssignVariableOpBassignvariableop_29_adam_sequential_2_sequential_conv2d_1_kernel_mIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30є
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_sequential_2_sequential_conv2d_1_bias_mIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31ї
AssignVariableOp_31AssignVariableOpBassignvariableop_31_adam_sequential_2_sequential_conv2d_2_kernel_mIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32є
AssignVariableOp_32AssignVariableOp@assignvariableop_32_adam_sequential_2_sequential_conv2d_2_bias_mIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33ї
AssignVariableOp_33AssignVariableOpBassignvariableop_33_adam_sequential_2_sequential_conv2d_3_kernel_mIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34є
AssignVariableOp_34AssignVariableOp@assignvariableop_34_adam_sequential_2_sequential_conv2d_3_bias_mIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35Є
AssignVariableOp_35AssignVariableOp?assignvariableop_35_adam_sequential_2_sequential_dense_kernel_mIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36ґ
AssignVariableOp_36AssignVariableOp=assignvariableop_36_adam_sequential_2_sequential_dense_bias_mIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37Љ
AssignVariableOp_37AssignVariableOpCassignvariableop_37_adam_sequential_2_sequential_1_dense_1_kernel_mIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38Ї
AssignVariableOp_38AssignVariableOpAassignvariableop_38_adam_sequential_2_sequential_1_dense_1_bias_mIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39≈
AssignVariableOp_39AssignVariableOpLassignvariableop_39_adam_sequential_2_sequential_1_conv2d_transpose_kernel_mIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40√
AssignVariableOp_40AssignVariableOpJassignvariableop_40_adam_sequential_2_sequential_1_conv2d_transpose_bias_mIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41«
AssignVariableOp_41AssignVariableOpNassignvariableop_41_adam_sequential_2_sequential_1_conv2d_transpose_1_kernel_mIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42≈
AssignVariableOp_42AssignVariableOpLassignvariableop_42_adam_sequential_2_sequential_1_conv2d_transpose_1_bias_mIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43«
AssignVariableOp_43AssignVariableOpNassignvariableop_43_adam_sequential_2_sequential_1_conv2d_transpose_2_kernel_mIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44≈
AssignVariableOp_44AssignVariableOpLassignvariableop_44_adam_sequential_2_sequential_1_conv2d_transpose_2_bias_mIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45«
AssignVariableOp_45AssignVariableOpNassignvariableop_45_adam_sequential_2_sequential_1_conv2d_transpose_3_kernel_mIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46≈
AssignVariableOp_46AssignVariableOpLassignvariableop_46_adam_sequential_2_sequential_1_conv2d_transpose_3_bias_mIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47є
AssignVariableOp_47AssignVariableOp@assignvariableop_47_adam_sequential_2_sequential_conv2d_kernel_vIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48Ј
AssignVariableOp_48AssignVariableOp>assignvariableop_48_adam_sequential_2_sequential_conv2d_bias_vIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49ї
AssignVariableOp_49AssignVariableOpBassignvariableop_49_adam_sequential_2_sequential_conv2d_1_kernel_vIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50є
AssignVariableOp_50AssignVariableOp@assignvariableop_50_adam_sequential_2_sequential_conv2d_1_bias_vIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51ї
AssignVariableOp_51AssignVariableOpBassignvariableop_51_adam_sequential_2_sequential_conv2d_2_kernel_vIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52є
AssignVariableOp_52AssignVariableOp@assignvariableop_52_adam_sequential_2_sequential_conv2d_2_bias_vIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53ї
AssignVariableOp_53AssignVariableOpBassignvariableop_53_adam_sequential_2_sequential_conv2d_3_kernel_vIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53_
Identity_54IdentityRestoreV2:tensors:54*
T0*
_output_shapes
:2
Identity_54є
AssignVariableOp_54AssignVariableOp@assignvariableop_54_adam_sequential_2_sequential_conv2d_3_bias_vIdentity_54:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_54_
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:2
Identity_55Є
AssignVariableOp_55AssignVariableOp?assignvariableop_55_adam_sequential_2_sequential_dense_kernel_vIdentity_55:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_55_
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:2
Identity_56ґ
AssignVariableOp_56AssignVariableOp=assignvariableop_56_adam_sequential_2_sequential_dense_bias_vIdentity_56:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_56_
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:2
Identity_57Љ
AssignVariableOp_57AssignVariableOpCassignvariableop_57_adam_sequential_2_sequential_1_dense_1_kernel_vIdentity_57:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_57_
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:2
Identity_58Ї
AssignVariableOp_58AssignVariableOpAassignvariableop_58_adam_sequential_2_sequential_1_dense_1_bias_vIdentity_58:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_58_
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:2
Identity_59≈
AssignVariableOp_59AssignVariableOpLassignvariableop_59_adam_sequential_2_sequential_1_conv2d_transpose_kernel_vIdentity_59:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_59_
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:2
Identity_60√
AssignVariableOp_60AssignVariableOpJassignvariableop_60_adam_sequential_2_sequential_1_conv2d_transpose_bias_vIdentity_60:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_60_
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:2
Identity_61«
AssignVariableOp_61AssignVariableOpNassignvariableop_61_adam_sequential_2_sequential_1_conv2d_transpose_1_kernel_vIdentity_61:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_61_
Identity_62IdentityRestoreV2:tensors:62*
T0*
_output_shapes
:2
Identity_62≈
AssignVariableOp_62AssignVariableOpLassignvariableop_62_adam_sequential_2_sequential_1_conv2d_transpose_1_bias_vIdentity_62:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_62_
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:2
Identity_63«
AssignVariableOp_63AssignVariableOpNassignvariableop_63_adam_sequential_2_sequential_1_conv2d_transpose_2_kernel_vIdentity_63:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_63_
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:2
Identity_64≈
AssignVariableOp_64AssignVariableOpLassignvariableop_64_adam_sequential_2_sequential_1_conv2d_transpose_2_bias_vIdentity_64:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_64_
Identity_65IdentityRestoreV2:tensors:65*
T0*
_output_shapes
:2
Identity_65«
AssignVariableOp_65AssignVariableOpNassignvariableop_65_adam_sequential_2_sequential_1_conv2d_transpose_3_kernel_vIdentity_65:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_65_
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:2
Identity_66≈
AssignVariableOp_66AssignVariableOpLassignvariableop_66_adam_sequential_2_sequential_1_conv2d_transpose_3_bias_vIdentity_66:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_66®
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesƒ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp†
Identity_67Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_67≠
Identity_68IdentityIdentity_67:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_68"#
identity_68Identity_68:output:0*£
_input_shapesС
О: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
в
∆
E__inference_sequential_layer_call_and_return_conditional_losses_36353

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐdense/StatefulPartitionedCallІ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€0@**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_361782 
conv2d/StatefulPartitionedCall“
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_361992"
 conv2d_1/StatefulPartitionedCall‘
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_362202"
 conv2d_2/StatefulPartitionedCall’
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€А**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_362412"
 conv2d_3/StatefulPartitionedCall≈
dense/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_362992
dense/StatefulPartitionedCallђ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:€€€€€€€€€`ј::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ѕ
±
0__inference_conv2d_transpose_layer_call_fn_36447

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_364392
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Є"
«
G__inference_sequential_1_layer_call_and_return_conditional_losses_36651
input_1*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_23
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_25
1conv2d_transpose_2_statefulpartitionedcall_args_15
1conv2d_transpose_2_statefulpartitionedcall_args_25
1conv2d_transpose_3_statefulpartitionedcall_args_15
1conv2d_transpose_3_statefulpartitionedcall_args_2
identityИҐ(conv2d_transpose/StatefulPartitionedCallҐ*conv2d_transpose_1/StatefulPartitionedCallҐ*conv2d_transpose_2/StatefulPartitionedCallҐ*conv2d_transpose_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallЃ
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_1&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€А**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_366262!
dense_1/StatefulPartitionedCallН
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_364392*
(conv2d_transpose/StatefulPartitionedCall†
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_364862,
*conv2d_transpose_1/StatefulPartitionedCallҐ
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:01conv2d_transpose_2_statefulpartitionedcall_args_11conv2d_transpose_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_365332,
*conv2d_transpose_2/StatefulPartitionedCallҐ
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:01conv2d_transpose_3_statefulpartitionedcall_args_11conv2d_transpose_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_365802,
*conv2d_transpose_3/StatefulPartitionedCallх
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
Б
ѕ
,__inference_sequential_1_layer_call_fn_36705
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_366922
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
£$
б
B__inference_dense_1_layer_call_and_return_conditional_losses_38094

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis—
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis„
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/ConstА
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1И
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackФ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЙ
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/transpose_1/permђ
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	А2
Tensordot/transpose_1З
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   А   2
Tensordot/Reshape_1/shapeЮ
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2
Tensordot/Reshape_1Ы
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Х
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
	TensordotН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpМ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2	
BiasAdd°
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
Э$
я
@__inference_dense_layer_call_and_return_conditional_losses_38053

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis—
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis„
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/ConstА
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1И
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackХ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЙ
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/transpose_1/permђ
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	А2
Tensordot/transpose_1З
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"А      2
Tensordot/Reshape_1/shapeЮ
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2
Tensordot/Reshape_1Ъ
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Ф
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЛ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd†
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
и
№
C__inference_conv2d_1_layer_call_and_return_conditional_losses_36199

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpґ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
©Р
ќ
G__inference_sequential_2_layer_call_and_return_conditional_losses_37217

inputs4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource6
2sequential_conv2d_3_conv2d_readvariableop_resource7
3sequential_conv2d_3_biasadd_readvariableop_resource6
2sequential_dense_tensordot_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource:
6sequential_1_dense_1_tensordot_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_3_biasadd_readvariableop_resource
identityИҐ(sequential/conv2d/BiasAdd/ReadVariableOpҐ'sequential/conv2d/Conv2D/ReadVariableOpҐ*sequential/conv2d_1/BiasAdd/ReadVariableOpҐ)sequential/conv2d_1/Conv2D/ReadVariableOpҐ*sequential/conv2d_2/BiasAdd/ReadVariableOpҐ)sequential/conv2d_2/Conv2D/ReadVariableOpҐ*sequential/conv2d_3/BiasAdd/ReadVariableOpҐ)sequential/conv2d_3/Conv2D/ReadVariableOpҐ'sequential/dense/BiasAdd/ReadVariableOpҐ)sequential/dense/Tensordot/ReadVariableOpҐ4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpҐ=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpҐ6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpҐ?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpҐ6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpҐ?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpҐ6sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOpҐ?sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpҐ+sequential_1/dense_1/BiasAdd/ReadVariableOpҐ-sequential_1/dense_1/Tensordot/ReadVariableOpЋ
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOpЏ
sequential/conv2d/Conv2DConv2Dinputs/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@*
paddingVALID*
strides
2
sequential/conv2d/Conv2D¬
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp–
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@2
sequential/conv2d/BiasAddЦ
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€0@2
sequential/conv2d/Relu—
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOpю
sequential/conv2d_1/Conv2DConv2D$sequential/conv2d/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D»
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOpЎ
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
sequential/conv2d_1/BiasAddЬ
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
sequential/conv2d_1/Relu—
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOpА
sequential/conv2d_2/Conv2DConv2D&sequential/conv2d_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2
sequential/conv2d_2/Conv2D»
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOpЎ
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
sequential/conv2d_2/BiasAddЬ
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
sequential/conv2d_2/Relu“
)sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02+
)sequential/conv2d_3/Conv2D/ReadVariableOpБ
sequential/conv2d_3/Conv2DConv2D&sequential/conv2d_2/Relu:activations:01sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
2
sequential/conv2d_3/Conv2D…
*sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*sequential/conv2d_3/BiasAdd/ReadVariableOpў
sequential/conv2d_3/BiasAddBiasAdd#sequential/conv2d_3/Conv2D:output:02sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential/conv2d_3/BiasAddЭ
sequential/conv2d_3/ReluRelu$sequential/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential/conv2d_3/Relu 
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype02+
)sequential/dense/Tensordot/ReadVariableOpМ
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
sequential/dense/Tensordot/axesЧ
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2!
sequential/dense/Tensordot/freeЪ
 sequential/dense/Tensordot/ShapeShape&sequential/conv2d_3/Relu:activations:0*
T0*
_output_shapes
:2"
 sequential/dense/Tensordot/ShapeЦ
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense/Tensordot/GatherV2/axis¶
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#sequential/dense/Tensordot/GatherV2Ъ
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense/Tensordot/GatherV2_1/axisђ
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%sequential/dense/Tensordot/GatherV2_1О
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 sequential/dense/Tensordot/Constƒ
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
sequential/dense/Tensordot/ProdТ
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/dense/Tensordot/Const_1ћ
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!sequential/dense/Tensordot/Prod_1Т
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/dense/Tensordot/concat/axisЕ
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!sequential/dense/Tensordot/concat–
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 sequential/dense/Tensordot/stackи
$sequential/dense/Tensordot/transpose	Transpose&sequential/conv2d_3/Relu:activations:0*sequential/dense/Tensordot/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2&
$sequential/dense/Tensordot/transposeг
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2$
"sequential/dense/Tensordot/ReshapeЂ
+sequential/dense/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2-
+sequential/dense/Tensordot/transpose_1/permр
&sequential/dense/Tensordot/transpose_1	Transpose1sequential/dense/Tensordot/ReadVariableOp:value:04sequential/dense/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	А2(
&sequential/dense/Tensordot/transpose_1©
*sequential/dense/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"А      2,
*sequential/dense/Tensordot/Reshape_1/shapeв
$sequential/dense/Tensordot/Reshape_1Reshape*sequential/dense/Tensordot/transpose_1:y:03sequential/dense/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2&
$sequential/dense/Tensordot/Reshape_1ё
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:0-sequential/dense/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€2#
!sequential/dense/Tensordot/MatMulТ
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/dense/Tensordot/Const_2Ц
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense/Tensordot/concat_1/axisТ
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense/Tensordot/concat_1Ў
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
sequential/dense/Tensordotњ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpѕ
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
sequential/dense/BiasAdd÷
-sequential_1/dense_1/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype02/
-sequential_1/dense_1/Tensordot/ReadVariableOpФ
#sequential_1/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_1/Tensordot/axesЯ
#sequential_1/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_1/dense_1/Tensordot/freeЭ
$sequential_1/dense_1/Tensordot/ShapeShape!sequential/dense/BiasAdd:output:0*
T0*
_output_shapes
:2&
$sequential_1/dense_1/Tensordot/ShapeЮ
,sequential_1/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_1/Tensordot/GatherV2/axisЇ
'sequential_1/dense_1/Tensordot/GatherV2GatherV2-sequential_1/dense_1/Tensordot/Shape:output:0,sequential_1/dense_1/Tensordot/free:output:05sequential_1/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_1/Tensordot/GatherV2Ґ
.sequential_1/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_1/Tensordot/GatherV2_1/axisј
)sequential_1/dense_1/Tensordot/GatherV2_1GatherV2-sequential_1/dense_1/Tensordot/Shape:output:0,sequential_1/dense_1/Tensordot/axes:output:07sequential_1/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_1/Tensordot/GatherV2_1Ц
$sequential_1/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_1/Tensordot/Const‘
#sequential_1/dense_1/Tensordot/ProdProd0sequential_1/dense_1/Tensordot/GatherV2:output:0-sequential_1/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_1/Tensordot/ProdЪ
&sequential_1/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_1/Tensordot/Const_1№
%sequential_1/dense_1/Tensordot/Prod_1Prod2sequential_1/dense_1/Tensordot/GatherV2_1:output:0/sequential_1/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_1/Tensordot/Prod_1Ъ
*sequential_1/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_1/Tensordot/concat/axisЩ
%sequential_1/dense_1/Tensordot/concatConcatV2,sequential_1/dense_1/Tensordot/free:output:0,sequential_1/dense_1/Tensordot/axes:output:03sequential_1/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_1/Tensordot/concatа
$sequential_1/dense_1/Tensordot/stackPack,sequential_1/dense_1/Tensordot/Prod:output:0.sequential_1/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_1/Tensordot/stackо
(sequential_1/dense_1/Tensordot/transpose	Transpose!sequential/dense/BiasAdd:output:0.sequential_1/dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€2*
(sequential_1/dense_1/Tensordot/transposeу
&sequential_1/dense_1/Tensordot/ReshapeReshape,sequential_1/dense_1/Tensordot/transpose:y:0-sequential_1/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_1/dense_1/Tensordot/Reshape≥
/sequential_1/dense_1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/sequential_1/dense_1/Tensordot/transpose_1/permА
*sequential_1/dense_1/Tensordot/transpose_1	Transpose5sequential_1/dense_1/Tensordot/ReadVariableOp:value:08sequential_1/dense_1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	А2,
*sequential_1/dense_1/Tensordot/transpose_1±
.sequential_1/dense_1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   А   20
.sequential_1/dense_1/Tensordot/Reshape_1/shapeт
(sequential_1/dense_1/Tensordot/Reshape_1Reshape.sequential_1/dense_1/Tensordot/transpose_1:y:07sequential_1/dense_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2*
(sequential_1/dense_1/Tensordot/Reshape_1п
%sequential_1/dense_1/Tensordot/MatMulMatMul/sequential_1/dense_1/Tensordot/Reshape:output:01sequential_1/dense_1/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%sequential_1/dense_1/Tensordot/MatMulЫ
&sequential_1/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2(
&sequential_1/dense_1/Tensordot/Const_2Ю
,sequential_1/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_1/Tensordot/concat_1/axis¶
'sequential_1/dense_1/Tensordot/concat_1ConcatV20sequential_1/dense_1/Tensordot/GatherV2:output:0/sequential_1/dense_1/Tensordot/Const_2:output:05sequential_1/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_1/Tensordot/concat_1й
sequential_1/dense_1/TensordotReshape/sequential_1/dense_1/Tensordot/MatMul:product:00sequential_1/dense_1/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
sequential_1/dense_1/Tensordotћ
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOpа
sequential_1/dense_1/BiasAddBiasAdd'sequential_1/dense_1/Tensordot:output:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_1/dense_1/BiasAddЯ
#sequential_1/conv2d_transpose/ShapeShape%sequential_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/Shape∞
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stackі
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1і
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2Ц
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_sliceі
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice_1/stackЄ
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Є
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2†
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/Shape:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1і
3sequential_1/conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice_2/stackЄ
5sequential_1/conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_2/stack_1Є
5sequential_1/conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_2/stack_2†
-sequential_1/conv2d_transpose/strided_slice_2StridedSlice,sequential_1/conv2d_transpose/Shape:output:0<sequential_1/conv2d_transpose/strided_slice_2/stack:output:0>sequential_1/conv2d_transpose/strided_slice_2/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_2М
#sequential_1/conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_1/conv2d_transpose/mul/y‘
!sequential_1/conv2d_transpose/mulMul6sequential_1/conv2d_transpose/strided_slice_1:output:0,sequential_1/conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_1/conv2d_transpose/mulМ
#sequential_1/conv2d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential_1/conv2d_transpose/add/y≈
!sequential_1/conv2d_transpose/addAddV2%sequential_1/conv2d_transpose/mul:z:0,sequential_1/conv2d_transpose/add/y:output:0*
T0*
_output_shapes
: 2#
!sequential_1/conv2d_transpose/addР
%sequential_1/conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/mul_1/yЏ
#sequential_1/conv2d_transpose/mul_1Mul6sequential_1/conv2d_transpose/strided_slice_2:output:0.sequential_1/conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose/mul_1Р
%sequential_1/conv2d_transpose/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/add_1/yЌ
#sequential_1/conv2d_transpose/add_1AddV2'sequential_1/conv2d_transpose/mul_1:z:0.sequential_1/conv2d_transpose/add_1/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose/add_1Р
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2'
%sequential_1/conv2d_transpose/stack/3ґ
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0%sequential_1/conv2d_transpose/add:z:0'sequential_1/conv2d_transpose/add_1:z:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stackі
3sequential_1/conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_3/stackЄ
5sequential_1/conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_3/stack_1Є
5sequential_1/conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_3/stack_2†
-sequential_1/conv2d_transpose/strided_slice_3StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_3/stack:output:0>sequential_1/conv2d_transpose/strided_slice_3/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_3О
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpц
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/dense_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
20
.sequential_1/conv2d_transpose/conv2d_transposeж
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpК
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2'
%sequential_1/conv2d_transpose/BiasAddЇ
"sequential_1/conv2d_transpose/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2$
"sequential_1/conv2d_transpose/ReluЃ
%sequential_1/conv2d_transpose_1/ShapeShape0sequential_1/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/Shapeі
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stackЄ
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Є
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Ґ
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_sliceЄ
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice_1/stackЉ
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Љ
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2ђ
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1Є
5sequential_1/conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice_2/stackЉ
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_1Љ
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_2ђ
/sequential_1/conv2d_transpose_1/strided_slice_2StridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0>sequential_1/conv2d_transpose_1/strided_slice_2/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_2/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_2Р
%sequential_1/conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose_1/mul/y№
#sequential_1/conv2d_transpose_1/mulMul8sequential_1/conv2d_transpose_1/strided_slice_1:output:0.sequential_1/conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_1/mulР
%sequential_1/conv2d_transpose_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose_1/add/yЌ
#sequential_1/conv2d_transpose_1/addAddV2'sequential_1/conv2d_transpose_1/mul:z:0.sequential_1/conv2d_transpose_1/add/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_1/addФ
'sequential_1/conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/mul_1/yв
%sequential_1/conv2d_transpose_1/mul_1Mul8sequential_1/conv2d_transpose_1/strided_slice_2:output:00sequential_1/conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_1/mul_1Ф
'sequential_1/conv2d_transpose_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/add_1/y’
%sequential_1/conv2d_transpose_1/add_1AddV2)sequential_1/conv2d_transpose_1/mul_1:z:00sequential_1/conv2d_transpose_1/add_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_1/add_1Ф
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_1/conv2d_transpose_1/stack/3¬
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:0'sequential_1/conv2d_transpose_1/add:z:0)sequential_1/conv2d_transpose_1/add_1:z:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stackЄ
5sequential_1/conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_3/stackЉ
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_1Љ
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_2ђ
/sequential_1/conv2d_transpose_1/strided_slice_3StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_3/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_3/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_3У
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpЙ
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:00sequential_1/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transposeм
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpТ
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2)
'sequential_1/conv2d_transpose_1/BiasAddј
$sequential_1/conv2d_transpose_1/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2&
$sequential_1/conv2d_transpose_1/Relu∞
%sequential_1/conv2d_transpose_2/ShapeShape2sequential_1/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/Shapeі
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stackЄ
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Є
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Ґ
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_sliceЄ
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice_1/stackЉ
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Љ
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2ђ
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1Є
5sequential_1/conv2d_transpose_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice_2/stackЉ
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_1Љ
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_2ђ
/sequential_1/conv2d_transpose_2/strided_slice_2StridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0>sequential_1/conv2d_transpose_2/strided_slice_2/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_2/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_2Р
%sequential_1/conv2d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose_2/mul/y№
#sequential_1/conv2d_transpose_2/mulMul8sequential_1/conv2d_transpose_2/strided_slice_1:output:0.sequential_1/conv2d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_2/mulР
%sequential_1/conv2d_transpose_2/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose_2/add/yЌ
#sequential_1/conv2d_transpose_2/addAddV2'sequential_1/conv2d_transpose_2/mul:z:0.sequential_1/conv2d_transpose_2/add/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_2/addФ
'sequential_1/conv2d_transpose_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/mul_1/yв
%sequential_1/conv2d_transpose_2/mul_1Mul8sequential_1/conv2d_transpose_2/strided_slice_2:output:00sequential_1/conv2d_transpose_2/mul_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_2/mul_1Ф
'sequential_1/conv2d_transpose_2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_2/add_1/y’
%sequential_1/conv2d_transpose_2/add_1AddV2)sequential_1/conv2d_transpose_2/mul_1:z:00sequential_1/conv2d_transpose_2/add_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_2/add_1Ф
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_1/conv2d_transpose_2/stack/3¬
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:0'sequential_1/conv2d_transpose_2/add:z:0)sequential_1/conv2d_transpose_2/add_1:z:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stackЄ
5sequential_1/conv2d_transpose_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_3/stackЉ
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_1Љ
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_2ђ
/sequential_1/conv2d_transpose_2/strided_slice_3StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_3/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_3/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_3У
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpЛ
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:02sequential_1/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€0@*
paddingVALID*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transposeм
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpТ
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@2)
'sequential_1/conv2d_transpose_2/BiasAddј
$sequential_1/conv2d_transpose_2/ReluRelu0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€0@2&
$sequential_1/conv2d_transpose_2/Relu∞
%sequential_1/conv2d_transpose_3/ShapeShape2sequential_1/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_3/Shapeі
3sequential_1/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_3/strided_slice/stackЄ
5sequential_1/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_3/strided_slice/stack_1Є
5sequential_1/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_3/strided_slice/stack_2Ґ
-sequential_1/conv2d_transpose_3/strided_sliceStridedSlice.sequential_1/conv2d_transpose_3/Shape:output:0<sequential_1/conv2d_transpose_3/strided_slice/stack:output:0>sequential_1/conv2d_transpose_3/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_3/strided_sliceЄ
5sequential_1/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_3/strided_slice_1/stackЉ
7sequential_1/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_3/strided_slice_1/stack_1Љ
7sequential_1/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_3/strided_slice_1/stack_2ђ
/sequential_1/conv2d_transpose_3/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_3/Shape:output:0>sequential_1/conv2d_transpose_3/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_3/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_3/strided_slice_1Є
5sequential_1/conv2d_transpose_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_3/strided_slice_2/stackЉ
7sequential_1/conv2d_transpose_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_3/strided_slice_2/stack_1Љ
7sequential_1/conv2d_transpose_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_3/strided_slice_2/stack_2ђ
/sequential_1/conv2d_transpose_3/strided_slice_2StridedSlice.sequential_1/conv2d_transpose_3/Shape:output:0>sequential_1/conv2d_transpose_3/strided_slice_2/stack:output:0@sequential_1/conv2d_transpose_3/strided_slice_2/stack_1:output:0@sequential_1/conv2d_transpose_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_3/strided_slice_2Р
%sequential_1/conv2d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose_3/mul/y№
#sequential_1/conv2d_transpose_3/mulMul8sequential_1/conv2d_transpose_3/strided_slice_1:output:0.sequential_1/conv2d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_3/mulР
%sequential_1/conv2d_transpose_3/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose_3/add/yЌ
#sequential_1/conv2d_transpose_3/addAddV2'sequential_1/conv2d_transpose_3/mul:z:0.sequential_1/conv2d_transpose_3/add/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_3/addФ
'sequential_1/conv2d_transpose_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_3/mul_1/yв
%sequential_1/conv2d_transpose_3/mul_1Mul8sequential_1/conv2d_transpose_3/strided_slice_2:output:00sequential_1/conv2d_transpose_3/mul_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_3/mul_1Ф
'sequential_1/conv2d_transpose_3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_3/add_1/y’
%sequential_1/conv2d_transpose_3/add_1AddV2)sequential_1/conv2d_transpose_3/mul_1:z:00sequential_1/conv2d_transpose_3/add_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_3/add_1Ф
'sequential_1/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_3/stack/3¬
%sequential_1/conv2d_transpose_3/stackPack6sequential_1/conv2d_transpose_3/strided_slice:output:0'sequential_1/conv2d_transpose_3/add:z:0)sequential_1/conv2d_transpose_3/add_1:z:00sequential_1/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_3/stackЄ
5sequential_1/conv2d_transpose_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_3/strided_slice_3/stackЉ
7sequential_1/conv2d_transpose_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_3/strided_slice_3/stack_1Љ
7sequential_1/conv2d_transpose_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_3/strided_slice_3/stack_2ђ
/sequential_1/conv2d_transpose_3/strided_slice_3StridedSlice.sequential_1/conv2d_transpose_3/stack:output:0>sequential_1/conv2d_transpose_3/strided_slice_3/stack:output:0@sequential_1/conv2d_transpose_3/strided_slice_3/stack_1:output:0@sequential_1/conv2d_transpose_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_3/strided_slice_3У
?sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02A
?sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpМ
0sequential_1/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_3/stack:output:0Gsequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:02sequential_1/conv2d_transpose_2/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€`ј*
paddingVALID*
strides
22
0sequential_1/conv2d_transpose_3/conv2d_transposeм
6sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOpУ
'sequential_1/conv2d_transpose_3/BiasAddBiasAdd9sequential_1/conv2d_transpose_3/conv2d_transpose:output:0>sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`ј2)
'sequential_1/conv2d_transpose_3/BiasAddЅ
$sequential_1/conv2d_transpose_3/ReluRelu0sequential_1/conv2d_transpose_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€`ј2&
$sequential_1/conv2d_transpose_3/ReluЛ	
IdentityIdentity2sequential_1/conv2d_transpose_3/Relu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp+^sequential/conv2d_3/BiasAdd/ReadVariableOp*^sequential/conv2d_3/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp5^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp.^sequential_1/dense_1/Tensordot/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€`ј2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€`ј::::::::::::::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2X
*sequential/conv2d_3/BiasAdd/ReadVariableOp*sequential/conv2d_3/BiasAdd/ReadVariableOp2V
)sequential/conv2d_3/Conv2D/ReadVariableOp)sequential/conv2d_3/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2В
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2В
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOp2В
?sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2^
-sequential_1/dense_1/Tensordot/ReadVariableOp-sequential_1/dense_1/Tensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
µ"
∆
G__inference_sequential_1_layer_call_and_return_conditional_losses_36726

inputs*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_23
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_25
1conv2d_transpose_2_statefulpartitionedcall_args_15
1conv2d_transpose_2_statefulpartitionedcall_args_25
1conv2d_transpose_3_statefulpartitionedcall_args_15
1conv2d_transpose_3_statefulpartitionedcall_args_2
identityИҐ(conv2d_transpose/StatefulPartitionedCallҐ*conv2d_transpose_1/StatefulPartitionedCallҐ*conv2d_transpose_2/StatefulPartitionedCallҐ*conv2d_transpose_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCall≠
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputs&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€А**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_366262!
dense_1/StatefulPartitionedCallН
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_364392*
(conv2d_transpose/StatefulPartitionedCall†
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_364862,
*conv2d_transpose_1/StatefulPartitionedCallҐ
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:01conv2d_transpose_2_statefulpartitionedcall_args_11conv2d_transpose_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_365332,
*conv2d_transpose_2/StatefulPartitionedCallҐ
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:01conv2d_transpose_3_statefulpartitionedcall_args_11conv2d_transpose_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_365802,
*conv2d_transpose_3/StatefulPartitionedCallх
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
ЮO
м
E__inference_sequential_layer_call_and_return_conditional_losses_37557

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpҐconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOpҐconv2d_3/BiasAdd/ReadVariableOpҐconv2d_3/Conv2D/ReadVariableOpҐdense/BiasAdd/ReadVariableOpҐdense/Tensordot/ReadVariableOp™
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOpє
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@*
paddingVALID*
strides
2
conv2d/Conv2D°
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp§
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€0@2
conv2d/Relu∞
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp“
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2
conv2d_1/Conv2DІ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_1/Relu∞
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp‘
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2
conv2d_2/Conv2DІ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpђ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_2/Relu±
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_3/Conv2D/ReadVariableOp’
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
2
conv2d_3/Conv2D®
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp≠
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_3/BiasAdd|
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_3/Relu©
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axesБ
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense/Tensordot/freey
dense/Tensordot/ShapeShapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
dense/Tensordot/ShapeА
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axisп
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2Д
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axisх
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/ConstШ
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1†
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axisќ
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat§
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stackЉ
dense/Tensordot/transpose	Transposeconv2d_3/Relu:activations:0dense/Tensordot/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dense/Tensordot/transposeЈ
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense/Tensordot/ReshapeХ
 dense/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/Tensordot/transpose_1/permƒ
dense/Tensordot/transpose_1	Transpose&dense/Tensordot/ReadVariableOp:value:0)dense/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	А2
dense/Tensordot/transpose_1У
dense/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"А      2!
dense/Tensordot/Reshape_1/shapeґ
dense/Tensordot/Reshape_1Reshapedense/Tensordot/transpose_1:y:0(dense/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2
dense/Tensordot/Reshape_1≤
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0"dense/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2А
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axisџ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1ђ
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
dense/TensordotЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp£
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
dense/BiasAddЇ
IdentityIdentitydense/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:€€€€€€€€€`ј::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
Њ
©
(__inference_conv2d_2_layer_call_fn_36228

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_362202
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
„
ћ
*__inference_sequential_layer_call_fn_37649

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИҐStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_363872
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:€€€€€€€€€`ј::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
н
ј
,__inference_sequential_2_layer_call_fn_37495

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20* 
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_369322
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€`ј::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
“
≥
2__inference_conv2d_transpose_1_layer_call_fn_36494

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_364862
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
„
ћ
*__inference_sequential_layer_call_fn_37634

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИҐStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_363532
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:€€€€€€€€€`ј::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ю
ќ
,__inference_sequential_1_layer_call_fn_38019

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_367262
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
–
ф
G__inference_sequential_2_layer_call_and_return_conditional_losses_36881

inputs-
)sequential_statefulpartitionedcall_args_1-
)sequential_statefulpartitionedcall_args_2-
)sequential_statefulpartitionedcall_args_3-
)sequential_statefulpartitionedcall_args_4-
)sequential_statefulpartitionedcall_args_5-
)sequential_statefulpartitionedcall_args_6-
)sequential_statefulpartitionedcall_args_7-
)sequential_statefulpartitionedcall_args_8-
)sequential_statefulpartitionedcall_args_9.
*sequential_statefulpartitionedcall_args_10/
+sequential_1_statefulpartitionedcall_args_1/
+sequential_1_statefulpartitionedcall_args_2/
+sequential_1_statefulpartitionedcall_args_3/
+sequential_1_statefulpartitionedcall_args_4/
+sequential_1_statefulpartitionedcall_args_5/
+sequential_1_statefulpartitionedcall_args_6/
+sequential_1_statefulpartitionedcall_args_7/
+sequential_1_statefulpartitionedcall_args_8/
+sequential_1_statefulpartitionedcall_args_90
,sequential_1_statefulpartitionedcall_args_10
identityИҐ"sequential/StatefulPartitionedCallҐ$sequential_1/StatefulPartitionedCallЬ
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputs)sequential_statefulpartitionedcall_args_1)sequential_statefulpartitionedcall_args_2)sequential_statefulpartitionedcall_args_3)sequential_statefulpartitionedcall_args_4)sequential_statefulpartitionedcall_args_5)sequential_statefulpartitionedcall_args_6)sequential_statefulpartitionedcall_args_7)sequential_statefulpartitionedcall_args_8)sequential_statefulpartitionedcall_args_9*sequential_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_363532$
"sequential/StatefulPartitionedCallн
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0+sequential_1_statefulpartitionedcall_args_1+sequential_1_statefulpartitionedcall_args_2+sequential_1_statefulpartitionedcall_args_3+sequential_1_statefulpartitionedcall_args_4+sequential_1_statefulpartitionedcall_args_5+sequential_1_statefulpartitionedcall_args_6+sequential_1_statefulpartitionedcall_args_7+sequential_1_statefulpartitionedcall_args_8+sequential_1_statefulpartitionedcall_args_9,sequential_1_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_366922&
$sequential_1/StatefulPartitionedCallз
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€`ј::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
р
Ѕ
,__inference_sequential_2_layer_call_fn_36904
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20* 
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_368812
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€`ј::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
“
≥
2__inference_conv2d_transpose_3_layer_call_fn_36588

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_365802
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Б
ѕ
,__inference_sequential_1_layer_call_fn_36739
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИҐStatefulPartitionedCall≠
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_367262
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
“
≥
2__inference_conv2d_transpose_2_layer_call_fn_36541

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_365332
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ј
©
(__inference_conv2d_3_layer_call_fn_36249

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCall†
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_362412
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
–
ф
G__inference_sequential_2_layer_call_and_return_conditional_losses_36932

inputs-
)sequential_statefulpartitionedcall_args_1-
)sequential_statefulpartitionedcall_args_2-
)sequential_statefulpartitionedcall_args_3-
)sequential_statefulpartitionedcall_args_4-
)sequential_statefulpartitionedcall_args_5-
)sequential_statefulpartitionedcall_args_6-
)sequential_statefulpartitionedcall_args_7-
)sequential_statefulpartitionedcall_args_8-
)sequential_statefulpartitionedcall_args_9.
*sequential_statefulpartitionedcall_args_10/
+sequential_1_statefulpartitionedcall_args_1/
+sequential_1_statefulpartitionedcall_args_2/
+sequential_1_statefulpartitionedcall_args_3/
+sequential_1_statefulpartitionedcall_args_4/
+sequential_1_statefulpartitionedcall_args_5/
+sequential_1_statefulpartitionedcall_args_6/
+sequential_1_statefulpartitionedcall_args_7/
+sequential_1_statefulpartitionedcall_args_8/
+sequential_1_statefulpartitionedcall_args_90
,sequential_1_statefulpartitionedcall_args_10
identityИҐ"sequential/StatefulPartitionedCallҐ$sequential_1/StatefulPartitionedCallЬ
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputs)sequential_statefulpartitionedcall_args_1)sequential_statefulpartitionedcall_args_2)sequential_statefulpartitionedcall_args_3)sequential_statefulpartitionedcall_args_4)sequential_statefulpartitionedcall_args_5)sequential_statefulpartitionedcall_args_6)sequential_statefulpartitionedcall_args_7)sequential_statefulpartitionedcall_args_8)sequential_statefulpartitionedcall_args_9*sequential_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_363872$
"sequential/StatefulPartitionedCallн
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0+sequential_1_statefulpartitionedcall_args_1+sequential_1_statefulpartitionedcall_args_2+sequential_1_statefulpartitionedcall_args_3+sequential_1_statefulpartitionedcall_args_4+sequential_1_statefulpartitionedcall_args_5+sequential_1_statefulpartitionedcall_args_6+sequential_1_statefulpartitionedcall_args_7+sequential_1_statefulpartitionedcall_args_8+sequential_1_statefulpartitionedcall_args_9,sequential_1_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_367262&
$sequential_1/StatefulPartitionedCallз
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€`ј::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
н&
ъ
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_36486

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2м
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3В
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3≥
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_transpose/ReadVariableOpс
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
Reluї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
Г
¶
%__inference_dense_layer_call_fn_38060

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_362992
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
в
∆
E__inference_sequential_layer_call_and_return_conditional_losses_36387

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐdense/StatefulPartitionedCallІ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€0@**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_361782 
conv2d/StatefulPartitionedCall“
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_361992"
 conv2d_1/StatefulPartitionedCall‘
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_362202"
 conv2d_2/StatefulPartitionedCall’
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€А**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_362412"
 conv2d_3/StatefulPartitionedCall≈
dense/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_362992
dense/StatefulPartitionedCallђ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:€€€€€€€€€`ј::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
Ю
Є
#__inference_signature_wrapper_36989
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identityИҐStatefulPartitionedCall…
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20* 
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€`ј**
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__wrapped_model_361652
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€`ј2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€`ј::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
Џ
Ќ
*__inference_sequential_layer_call_fn_36366
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_363532
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:€€€€€€€€€`ј::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
о
№
C__inference_conv2d_3_layer_call_and_return_conditional_losses_36241

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateЦ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02
Conv2D/ReadVariableOpЈ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*
paddingVALID*
strides
2
Conv2DН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЫ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2	
BiasAdds
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2
Relu≤
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
©Р
ќ
G__inference_sequential_2_layer_call_and_return_conditional_losses_37445

inputs4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource6
2sequential_conv2d_3_conv2d_readvariableop_resource7
3sequential_conv2d_3_biasadd_readvariableop_resource6
2sequential_dense_tensordot_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource:
6sequential_1_dense_1_tensordot_readvariableop_resource8
4sequential_1_dense_1_biasadd_readvariableop_resourceJ
Fsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceA
=sequential_1_conv2d_transpose_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resourceL
Hsequential_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resourceC
?sequential_1_conv2d_transpose_3_biasadd_readvariableop_resource
identityИҐ(sequential/conv2d/BiasAdd/ReadVariableOpҐ'sequential/conv2d/Conv2D/ReadVariableOpҐ*sequential/conv2d_1/BiasAdd/ReadVariableOpҐ)sequential/conv2d_1/Conv2D/ReadVariableOpҐ*sequential/conv2d_2/BiasAdd/ReadVariableOpҐ)sequential/conv2d_2/Conv2D/ReadVariableOpҐ*sequential/conv2d_3/BiasAdd/ReadVariableOpҐ)sequential/conv2d_3/Conv2D/ReadVariableOpҐ'sequential/dense/BiasAdd/ReadVariableOpҐ)sequential/dense/Tensordot/ReadVariableOpҐ4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpҐ=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpҐ6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpҐ?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpҐ6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpҐ?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpҐ6sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOpҐ?sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpҐ+sequential_1/dense_1/BiasAdd/ReadVariableOpҐ-sequential_1/dense_1/Tensordot/ReadVariableOpЋ
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOpЏ
sequential/conv2d/Conv2DConv2Dinputs/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@*
paddingVALID*
strides
2
sequential/conv2d/Conv2D¬
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp–
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@2
sequential/conv2d/BiasAddЦ
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€0@2
sequential/conv2d/Relu—
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOpю
sequential/conv2d_1/Conv2DConv2D$sequential/conv2d/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D»
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOpЎ
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
sequential/conv2d_1/BiasAddЬ
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
sequential/conv2d_1/Relu—
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02+
)sequential/conv2d_2/Conv2D/ReadVariableOpА
sequential/conv2d_2/Conv2DConv2D&sequential/conv2d_1/Relu:activations:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2
sequential/conv2d_2/Conv2D»
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*sequential/conv2d_2/BiasAdd/ReadVariableOpЎ
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
sequential/conv2d_2/BiasAddЬ
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
sequential/conv2d_2/Relu“
)sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02+
)sequential/conv2d_3/Conv2D/ReadVariableOpБ
sequential/conv2d_3/Conv2DConv2D&sequential/conv2d_2/Relu:activations:01sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
2
sequential/conv2d_3/Conv2D…
*sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02,
*sequential/conv2d_3/BiasAdd/ReadVariableOpў
sequential/conv2d_3/BiasAddBiasAdd#sequential/conv2d_3/Conv2D:output:02sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential/conv2d_3/BiasAddЭ
sequential/conv2d_3/ReluRelu$sequential/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential/conv2d_3/Relu 
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype02+
)sequential/dense/Tensordot/ReadVariableOpМ
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
sequential/dense/Tensordot/axesЧ
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2!
sequential/dense/Tensordot/freeЪ
 sequential/dense/Tensordot/ShapeShape&sequential/conv2d_3/Relu:activations:0*
T0*
_output_shapes
:2"
 sequential/dense/Tensordot/ShapeЦ
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense/Tensordot/GatherV2/axis¶
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#sequential/dense/Tensordot/GatherV2Ъ
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense/Tensordot/GatherV2_1/axisђ
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%sequential/dense/Tensordot/GatherV2_1О
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 sequential/dense/Tensordot/Constƒ
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
sequential/dense/Tensordot/ProdТ
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/dense/Tensordot/Const_1ћ
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!sequential/dense/Tensordot/Prod_1Т
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/dense/Tensordot/concat/axisЕ
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!sequential/dense/Tensordot/concat–
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 sequential/dense/Tensordot/stackи
$sequential/dense/Tensordot/transpose	Transpose&sequential/conv2d_3/Relu:activations:0*sequential/dense/Tensordot/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2&
$sequential/dense/Tensordot/transposeг
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2$
"sequential/dense/Tensordot/ReshapeЂ
+sequential/dense/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2-
+sequential/dense/Tensordot/transpose_1/permр
&sequential/dense/Tensordot/transpose_1	Transpose1sequential/dense/Tensordot/ReadVariableOp:value:04sequential/dense/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	А2(
&sequential/dense/Tensordot/transpose_1©
*sequential/dense/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"А      2,
*sequential/dense/Tensordot/Reshape_1/shapeв
$sequential/dense/Tensordot/Reshape_1Reshape*sequential/dense/Tensordot/transpose_1:y:03sequential/dense/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2&
$sequential/dense/Tensordot/Reshape_1ё
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:0-sequential/dense/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€2#
!sequential/dense/Tensordot/MatMulТ
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/dense/Tensordot/Const_2Ц
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense/Tensordot/concat_1/axisТ
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense/Tensordot/concat_1Ў
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
sequential/dense/Tensordotњ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpѕ
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
sequential/dense/BiasAdd÷
-sequential_1/dense_1/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype02/
-sequential_1/dense_1/Tensordot/ReadVariableOpФ
#sequential_1/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_1/dense_1/Tensordot/axesЯ
#sequential_1/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_1/dense_1/Tensordot/freeЭ
$sequential_1/dense_1/Tensordot/ShapeShape!sequential/dense/BiasAdd:output:0*
T0*
_output_shapes
:2&
$sequential_1/dense_1/Tensordot/ShapeЮ
,sequential_1/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_1/Tensordot/GatherV2/axisЇ
'sequential_1/dense_1/Tensordot/GatherV2GatherV2-sequential_1/dense_1/Tensordot/Shape:output:0,sequential_1/dense_1/Tensordot/free:output:05sequential_1/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_1/dense_1/Tensordot/GatherV2Ґ
.sequential_1/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_1/dense_1/Tensordot/GatherV2_1/axisј
)sequential_1/dense_1/Tensordot/GatherV2_1GatherV2-sequential_1/dense_1/Tensordot/Shape:output:0,sequential_1/dense_1/Tensordot/axes:output:07sequential_1/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_1/dense_1/Tensordot/GatherV2_1Ц
$sequential_1/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_1/dense_1/Tensordot/Const‘
#sequential_1/dense_1/Tensordot/ProdProd0sequential_1/dense_1/Tensordot/GatherV2:output:0-sequential_1/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_1/dense_1/Tensordot/ProdЪ
&sequential_1/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_1/dense_1/Tensordot/Const_1№
%sequential_1/dense_1/Tensordot/Prod_1Prod2sequential_1/dense_1/Tensordot/GatherV2_1:output:0/sequential_1/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_1/dense_1/Tensordot/Prod_1Ъ
*sequential_1/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_1/dense_1/Tensordot/concat/axisЩ
%sequential_1/dense_1/Tensordot/concatConcatV2,sequential_1/dense_1/Tensordot/free:output:0,sequential_1/dense_1/Tensordot/axes:output:03sequential_1/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/dense_1/Tensordot/concatа
$sequential_1/dense_1/Tensordot/stackPack,sequential_1/dense_1/Tensordot/Prod:output:0.sequential_1/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_1/dense_1/Tensordot/stackо
(sequential_1/dense_1/Tensordot/transpose	Transpose!sequential/dense/BiasAdd:output:0.sequential_1/dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€2*
(sequential_1/dense_1/Tensordot/transposeу
&sequential_1/dense_1/Tensordot/ReshapeReshape,sequential_1/dense_1/Tensordot/transpose:y:0-sequential_1/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2(
&sequential_1/dense_1/Tensordot/Reshape≥
/sequential_1/dense_1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       21
/sequential_1/dense_1/Tensordot/transpose_1/permА
*sequential_1/dense_1/Tensordot/transpose_1	Transpose5sequential_1/dense_1/Tensordot/ReadVariableOp:value:08sequential_1/dense_1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	А2,
*sequential_1/dense_1/Tensordot/transpose_1±
.sequential_1/dense_1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   А   20
.sequential_1/dense_1/Tensordot/Reshape_1/shapeт
(sequential_1/dense_1/Tensordot/Reshape_1Reshape.sequential_1/dense_1/Tensordot/transpose_1:y:07sequential_1/dense_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2*
(sequential_1/dense_1/Tensordot/Reshape_1п
%sequential_1/dense_1/Tensordot/MatMulMatMul/sequential_1/dense_1/Tensordot/Reshape:output:01sequential_1/dense_1/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2'
%sequential_1/dense_1/Tensordot/MatMulЫ
&sequential_1/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2(
&sequential_1/dense_1/Tensordot/Const_2Ю
,sequential_1/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_1/dense_1/Tensordot/concat_1/axis¶
'sequential_1/dense_1/Tensordot/concat_1ConcatV20sequential_1/dense_1/Tensordot/GatherV2:output:0/sequential_1/dense_1/Tensordot/Const_2:output:05sequential_1/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_1/dense_1/Tensordot/concat_1й
sequential_1/dense_1/TensordotReshape/sequential_1/dense_1/Tensordot/MatMul:product:00sequential_1/dense_1/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2 
sequential_1/dense_1/Tensordotћ
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+sequential_1/dense_1/BiasAdd/ReadVariableOpа
sequential_1/dense_1/BiasAddBiasAdd'sequential_1/dense_1/Tensordot:output:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
sequential_1/dense_1/BiasAddЯ
#sequential_1/conv2d_transpose/ShapeShape%sequential_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/Shape∞
1sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_1/conv2d_transpose/strided_slice/stackі
3sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_1і
3sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice/stack_2Ц
+sequential_1/conv2d_transpose/strided_sliceStridedSlice,sequential_1/conv2d_transpose/Shape:output:0:sequential_1/conv2d_transpose/strided_slice/stack:output:0<sequential_1/conv2d_transpose/strided_slice/stack_1:output:0<sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+sequential_1/conv2d_transpose/strided_sliceі
3sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice_1/stackЄ
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_1Є
5sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_1/stack_2†
-sequential_1/conv2d_transpose/strided_slice_1StridedSlice,sequential_1/conv2d_transpose/Shape:output:0<sequential_1/conv2d_transpose/strided_slice_1/stack:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_1і
3sequential_1/conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:25
3sequential_1/conv2d_transpose/strided_slice_2/stackЄ
5sequential_1/conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_2/stack_1Є
5sequential_1/conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_2/stack_2†
-sequential_1/conv2d_transpose/strided_slice_2StridedSlice,sequential_1/conv2d_transpose/Shape:output:0<sequential_1/conv2d_transpose/strided_slice_2/stack:output:0>sequential_1/conv2d_transpose/strided_slice_2/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_2М
#sequential_1/conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_1/conv2d_transpose/mul/y‘
!sequential_1/conv2d_transpose/mulMul6sequential_1/conv2d_transpose/strided_slice_1:output:0,sequential_1/conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_1/conv2d_transpose/mulМ
#sequential_1/conv2d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2%
#sequential_1/conv2d_transpose/add/y≈
!sequential_1/conv2d_transpose/addAddV2%sequential_1/conv2d_transpose/mul:z:0,sequential_1/conv2d_transpose/add/y:output:0*
T0*
_output_shapes
: 2#
!sequential_1/conv2d_transpose/addР
%sequential_1/conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose/mul_1/yЏ
#sequential_1/conv2d_transpose/mul_1Mul6sequential_1/conv2d_transpose/strided_slice_2:output:0.sequential_1/conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose/mul_1Р
%sequential_1/conv2d_transpose/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose/add_1/yЌ
#sequential_1/conv2d_transpose/add_1AddV2'sequential_1/conv2d_transpose/mul_1:z:0.sequential_1/conv2d_transpose/add_1/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose/add_1Р
%sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2'
%sequential_1/conv2d_transpose/stack/3ґ
#sequential_1/conv2d_transpose/stackPack4sequential_1/conv2d_transpose/strided_slice:output:0%sequential_1/conv2d_transpose/add:z:0'sequential_1/conv2d_transpose/add_1:z:0.sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2%
#sequential_1/conv2d_transpose/stackі
3sequential_1/conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose/strided_slice_3/stackЄ
5sequential_1/conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_3/stack_1Є
5sequential_1/conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose/strided_slice_3/stack_2†
-sequential_1/conv2d_transpose/strided_slice_3StridedSlice,sequential_1/conv2d_transpose/stack:output:0<sequential_1/conv2d_transpose/strided_slice_3/stack:output:0>sequential_1/conv2d_transpose/strided_slice_3/stack_1:output:0>sequential_1/conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose/strided_slice_3О
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpFsequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype02?
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpц
.sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput,sequential_1/conv2d_transpose/stack:output:0Esequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0%sequential_1/dense_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
20
.sequential_1/conv2d_transpose/conv2d_transposeж
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp=sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpК
%sequential_1/conv2d_transpose/BiasAddBiasAdd7sequential_1/conv2d_transpose/conv2d_transpose:output:0<sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2'
%sequential_1/conv2d_transpose/BiasAddЇ
"sequential_1/conv2d_transpose/ReluRelu.sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2$
"sequential_1/conv2d_transpose/ReluЃ
%sequential_1/conv2d_transpose_1/ShapeShape0sequential_1/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/Shapeі
3sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_1/strided_slice/stackЄ
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_1Є
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice/stack_2Ґ
-sequential_1/conv2d_transpose_1/strided_sliceStridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0<sequential_1/conv2d_transpose_1/strided_slice/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_1/strided_sliceЄ
5sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice_1/stackЉ
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Љ
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_1/stack_2ђ
/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0>sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_1Є
5sequential_1/conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_1/strided_slice_2/stackЉ
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_1Љ
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_2/stack_2ђ
/sequential_1/conv2d_transpose_1/strided_slice_2StridedSlice.sequential_1/conv2d_transpose_1/Shape:output:0>sequential_1/conv2d_transpose_1/strided_slice_2/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_2/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_2Р
%sequential_1/conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose_1/mul/y№
#sequential_1/conv2d_transpose_1/mulMul8sequential_1/conv2d_transpose_1/strided_slice_1:output:0.sequential_1/conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_1/mulР
%sequential_1/conv2d_transpose_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose_1/add/yЌ
#sequential_1/conv2d_transpose_1/addAddV2'sequential_1/conv2d_transpose_1/mul:z:0.sequential_1/conv2d_transpose_1/add/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_1/addФ
'sequential_1/conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_1/mul_1/yв
%sequential_1/conv2d_transpose_1/mul_1Mul8sequential_1/conv2d_transpose_1/strided_slice_2:output:00sequential_1/conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_1/mul_1Ф
'sequential_1/conv2d_transpose_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_1/add_1/y’
%sequential_1/conv2d_transpose_1/add_1AddV2)sequential_1/conv2d_transpose_1/mul_1:z:00sequential_1/conv2d_transpose_1/add_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_1/add_1Ф
'sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_1/conv2d_transpose_1/stack/3¬
%sequential_1/conv2d_transpose_1/stackPack6sequential_1/conv2d_transpose_1/strided_slice:output:0'sequential_1/conv2d_transpose_1/add:z:0)sequential_1/conv2d_transpose_1/add_1:z:00sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_1/stackЄ
5sequential_1/conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_1/strided_slice_3/stackЉ
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_1Љ
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_1/strided_slice_3/stack_2ђ
/sequential_1/conv2d_transpose_1/strided_slice_3StridedSlice.sequential_1/conv2d_transpose_1/stack:output:0>sequential_1/conv2d_transpose_1/strided_slice_3/stack:output:0@sequential_1/conv2d_transpose_1/strided_slice_3/stack_1:output:0@sequential_1/conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_1/strided_slice_3У
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02A
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpЙ
0sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_1/stack:output:0Gsequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:00sequential_1/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
22
0sequential_1/conv2d_transpose_1/conv2d_transposeм
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpТ
'sequential_1/conv2d_transpose_1/BiasAddBiasAdd9sequential_1/conv2d_transpose_1/conv2d_transpose:output:0>sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2)
'sequential_1/conv2d_transpose_1/BiasAddј
$sequential_1/conv2d_transpose_1/ReluRelu0sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2&
$sequential_1/conv2d_transpose_1/Relu∞
%sequential_1/conv2d_transpose_2/ShapeShape2sequential_1/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/Shapeі
3sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_2/strided_slice/stackЄ
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_1Є
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice/stack_2Ґ
-sequential_1/conv2d_transpose_2/strided_sliceStridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0<sequential_1/conv2d_transpose_2/strided_slice/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_2/strided_sliceЄ
5sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice_1/stackЉ
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Љ
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_1/stack_2ђ
/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0>sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_1Є
5sequential_1/conv2d_transpose_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_2/strided_slice_2/stackЉ
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_1Љ
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_2/stack_2ђ
/sequential_1/conv2d_transpose_2/strided_slice_2StridedSlice.sequential_1/conv2d_transpose_2/Shape:output:0>sequential_1/conv2d_transpose_2/strided_slice_2/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_2/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_2Р
%sequential_1/conv2d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose_2/mul/y№
#sequential_1/conv2d_transpose_2/mulMul8sequential_1/conv2d_transpose_2/strided_slice_1:output:0.sequential_1/conv2d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_2/mulР
%sequential_1/conv2d_transpose_2/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose_2/add/yЌ
#sequential_1/conv2d_transpose_2/addAddV2'sequential_1/conv2d_transpose_2/mul:z:0.sequential_1/conv2d_transpose_2/add/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_2/addФ
'sequential_1/conv2d_transpose_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_2/mul_1/yв
%sequential_1/conv2d_transpose_2/mul_1Mul8sequential_1/conv2d_transpose_2/strided_slice_2:output:00sequential_1/conv2d_transpose_2/mul_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_2/mul_1Ф
'sequential_1/conv2d_transpose_2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_2/add_1/y’
%sequential_1/conv2d_transpose_2/add_1AddV2)sequential_1/conv2d_transpose_2/mul_1:z:00sequential_1/conv2d_transpose_2/add_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_2/add_1Ф
'sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2)
'sequential_1/conv2d_transpose_2/stack/3¬
%sequential_1/conv2d_transpose_2/stackPack6sequential_1/conv2d_transpose_2/strided_slice:output:0'sequential_1/conv2d_transpose_2/add:z:0)sequential_1/conv2d_transpose_2/add_1:z:00sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_2/stackЄ
5sequential_1/conv2d_transpose_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_2/strided_slice_3/stackЉ
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_1Љ
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_2/strided_slice_3/stack_2ђ
/sequential_1/conv2d_transpose_2/strided_slice_3StridedSlice.sequential_1/conv2d_transpose_2/stack:output:0>sequential_1/conv2d_transpose_2/strided_slice_3/stack:output:0@sequential_1/conv2d_transpose_2/strided_slice_3/stack_1:output:0@sequential_1/conv2d_transpose_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_2/strided_slice_3У
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02A
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpЛ
0sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_2/stack:output:0Gsequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:02sequential_1/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€0@*
paddingVALID*
strides
22
0sequential_1/conv2d_transpose_2/conv2d_transposeм
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype028
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpТ
'sequential_1/conv2d_transpose_2/BiasAddBiasAdd9sequential_1/conv2d_transpose_2/conv2d_transpose:output:0>sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@2)
'sequential_1/conv2d_transpose_2/BiasAddј
$sequential_1/conv2d_transpose_2/ReluRelu0sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€0@2&
$sequential_1/conv2d_transpose_2/Relu∞
%sequential_1/conv2d_transpose_3/ShapeShape2sequential_1/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_3/Shapeі
3sequential_1/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/conv2d_transpose_3/strided_slice/stackЄ
5sequential_1/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_3/strided_slice/stack_1Є
5sequential_1/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_3/strided_slice/stack_2Ґ
-sequential_1/conv2d_transpose_3/strided_sliceStridedSlice.sequential_1/conv2d_transpose_3/Shape:output:0<sequential_1/conv2d_transpose_3/strided_slice/stack:output:0>sequential_1/conv2d_transpose_3/strided_slice/stack_1:output:0>sequential_1/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-sequential_1/conv2d_transpose_3/strided_sliceЄ
5sequential_1/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_3/strided_slice_1/stackЉ
7sequential_1/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_3/strided_slice_1/stack_1Љ
7sequential_1/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_3/strided_slice_1/stack_2ђ
/sequential_1/conv2d_transpose_3/strided_slice_1StridedSlice.sequential_1/conv2d_transpose_3/Shape:output:0>sequential_1/conv2d_transpose_3/strided_slice_1/stack:output:0@sequential_1/conv2d_transpose_3/strided_slice_1/stack_1:output:0@sequential_1/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_3/strided_slice_1Є
5sequential_1/conv2d_transpose_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5sequential_1/conv2d_transpose_3/strided_slice_2/stackЉ
7sequential_1/conv2d_transpose_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_3/strided_slice_2/stack_1Љ
7sequential_1/conv2d_transpose_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_3/strided_slice_2/stack_2ђ
/sequential_1/conv2d_transpose_3/strided_slice_2StridedSlice.sequential_1/conv2d_transpose_3/Shape:output:0>sequential_1/conv2d_transpose_3/strided_slice_2/stack:output:0@sequential_1/conv2d_transpose_3/strided_slice_2/stack_1:output:0@sequential_1/conv2d_transpose_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_3/strided_slice_2Р
%sequential_1/conv2d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential_1/conv2d_transpose_3/mul/y№
#sequential_1/conv2d_transpose_3/mulMul8sequential_1/conv2d_transpose_3/strided_slice_1:output:0.sequential_1/conv2d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_3/mulР
%sequential_1/conv2d_transpose_3/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2'
%sequential_1/conv2d_transpose_3/add/yЌ
#sequential_1/conv2d_transpose_3/addAddV2'sequential_1/conv2d_transpose_3/mul:z:0.sequential_1/conv2d_transpose_3/add/y:output:0*
T0*
_output_shapes
: 2%
#sequential_1/conv2d_transpose_3/addФ
'sequential_1/conv2d_transpose_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_3/mul_1/yв
%sequential_1/conv2d_transpose_3/mul_1Mul8sequential_1/conv2d_transpose_3/strided_slice_2:output:00sequential_1/conv2d_transpose_3/mul_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_3/mul_1Ф
'sequential_1/conv2d_transpose_3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_1/conv2d_transpose_3/add_1/y’
%sequential_1/conv2d_transpose_3/add_1AddV2)sequential_1/conv2d_transpose_3/mul_1:z:00sequential_1/conv2d_transpose_3/add_1/y:output:0*
T0*
_output_shapes
: 2'
%sequential_1/conv2d_transpose_3/add_1Ф
'sequential_1/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential_1/conv2d_transpose_3/stack/3¬
%sequential_1/conv2d_transpose_3/stackPack6sequential_1/conv2d_transpose_3/strided_slice:output:0'sequential_1/conv2d_transpose_3/add:z:0)sequential_1/conv2d_transpose_3/add_1:z:00sequential_1/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2'
%sequential_1/conv2d_transpose_3/stackЄ
5sequential_1/conv2d_transpose_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_1/conv2d_transpose_3/strided_slice_3/stackЉ
7sequential_1/conv2d_transpose_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_3/strided_slice_3/stack_1Љ
7sequential_1/conv2d_transpose_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_1/conv2d_transpose_3/strided_slice_3/stack_2ђ
/sequential_1/conv2d_transpose_3/strided_slice_3StridedSlice.sequential_1/conv2d_transpose_3/stack:output:0>sequential_1/conv2d_transpose_3/strided_slice_3/stack:output:0@sequential_1/conv2d_transpose_3/strided_slice_3/stack_1:output:0@sequential_1/conv2d_transpose_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_1/conv2d_transpose_3/strided_slice_3У
?sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpHsequential_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02A
?sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpМ
0sequential_1/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput.sequential_1/conv2d_transpose_3/stack:output:0Gsequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:02sequential_1/conv2d_transpose_2/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€`ј*
paddingVALID*
strides
22
0sequential_1/conv2d_transpose_3/conv2d_transposeм
6sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOpУ
'sequential_1/conv2d_transpose_3/BiasAddBiasAdd9sequential_1/conv2d_transpose_3/conv2d_transpose:output:0>sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`ј2)
'sequential_1/conv2d_transpose_3/BiasAddЅ
$sequential_1/conv2d_transpose_3/ReluRelu0sequential_1/conv2d_transpose_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€`ј2&
$sequential_1/conv2d_transpose_3/ReluЛ	
IdentityIdentity2sequential_1/conv2d_transpose_3/Relu:activations:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp+^sequential/conv2d_3/BiasAdd/ReadVariableOp*^sequential/conv2d_3/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/Tensordot/ReadVariableOp5^sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp>^sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp7^sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOp@^sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp.^sequential_1/dense_1/Tensordot/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€`ј2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€`ј::::::::::::::::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2X
*sequential/conv2d_3/BiasAdd/ReadVariableOp*sequential/conv2d_3/BiasAdd/ReadVariableOp2V
)sequential/conv2d_3/Conv2D/ReadVariableOp)sequential/conv2d_3/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/Tensordot/ReadVariableOp)sequential/dense/Tensordot/ReadVariableOp2l
4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp4sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2~
=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp=sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2В
?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2В
?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2p
6sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOp6sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOp2В
?sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2^
-sequential_1/dense_1/Tensordot/ReadVariableOp-sequential_1/dense_1/Tensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
”
х
G__inference_sequential_2_layer_call_and_return_conditional_losses_36852
input_1-
)sequential_statefulpartitionedcall_args_1-
)sequential_statefulpartitionedcall_args_2-
)sequential_statefulpartitionedcall_args_3-
)sequential_statefulpartitionedcall_args_4-
)sequential_statefulpartitionedcall_args_5-
)sequential_statefulpartitionedcall_args_6-
)sequential_statefulpartitionedcall_args_7-
)sequential_statefulpartitionedcall_args_8-
)sequential_statefulpartitionedcall_args_9.
*sequential_statefulpartitionedcall_args_10/
+sequential_1_statefulpartitionedcall_args_1/
+sequential_1_statefulpartitionedcall_args_2/
+sequential_1_statefulpartitionedcall_args_3/
+sequential_1_statefulpartitionedcall_args_4/
+sequential_1_statefulpartitionedcall_args_5/
+sequential_1_statefulpartitionedcall_args_6/
+sequential_1_statefulpartitionedcall_args_7/
+sequential_1_statefulpartitionedcall_args_8/
+sequential_1_statefulpartitionedcall_args_90
,sequential_1_statefulpartitionedcall_args_10
identityИҐ"sequential/StatefulPartitionedCallҐ$sequential_1/StatefulPartitionedCallЭ
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1)sequential_statefulpartitionedcall_args_1)sequential_statefulpartitionedcall_args_2)sequential_statefulpartitionedcall_args_3)sequential_statefulpartitionedcall_args_4)sequential_statefulpartitionedcall_args_5)sequential_statefulpartitionedcall_args_6)sequential_statefulpartitionedcall_args_7)sequential_statefulpartitionedcall_args_8)sequential_statefulpartitionedcall_args_9*sequential_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_363872$
"sequential/StatefulPartitionedCallн
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0+sequential_1_statefulpartitionedcall_args_1+sequential_1_statefulpartitionedcall_args_2+sequential_1_statefulpartitionedcall_args_3+sequential_1_statefulpartitionedcall_args_4+sequential_1_statefulpartitionedcall_args_5+sequential_1_statefulpartitionedcall_args_6+sequential_1_statefulpartitionedcall_args_7+sequential_1_statefulpartitionedcall_args_8+sequential_1_statefulpartitionedcall_args_9,sequential_1_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_367262&
$sequential_1/StatefulPartitionedCallз
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€`ј::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
ЮO
м
E__inference_sequential_layer_call_and_return_conditional_losses_37619

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИҐconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpҐconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOpҐconv2d_3/BiasAdd/ReadVariableOpҐconv2d_3/Conv2D/ReadVariableOpҐdense/BiasAdd/ReadVariableOpҐdense/Tensordot/ReadVariableOp™
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOpє
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@*
paddingVALID*
strides
2
conv2d/Conv2D°
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp§
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€0@2
conv2d/Relu∞
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp“
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2
conv2d_1/Conv2DІ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_1/Relu∞
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOp‘
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2
conv2d_2/Conv2DІ
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOpђ
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_2/Relu±
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02 
conv2d_3/Conv2D/ReadVariableOp’
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
2
conv2d_3/Conv2D®
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp≠
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_3/BiasAdd|
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
conv2d_3/Relu©
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axesБ
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense/Tensordot/freey
dense/Tensordot/ShapeShapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
dense/Tensordot/ShapeА
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axisп
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2Д
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axisх
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/ConstШ
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1†
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axisќ
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat§
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stackЉ
dense/Tensordot/transpose	Transposeconv2d_3/Relu:activations:0dense/Tensordot/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dense/Tensordot/transposeЈ
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense/Tensordot/ReshapeХ
 dense/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2"
 dense/Tensordot/transpose_1/permƒ
dense/Tensordot/transpose_1	Transpose&dense/Tensordot/ReadVariableOp:value:0)dense/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	А2
dense/Tensordot/transpose_1У
dense/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"А      2!
dense/Tensordot/Reshape_1/shapeґ
dense/Tensordot/Reshape_1Reshapedense/Tensordot/transpose_1:y:0(dense/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2
dense/Tensordot/Reshape_1≤
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0"dense/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2А
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axisџ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1ђ
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
dense/TensordotЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp£
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
dense/BiasAddЇ
IdentityIdentitydense/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:€€€€€€€€€`ј::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
ю
ќ
,__inference_sequential_1_layer_call_fn_38004

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_366922
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Э$
я
@__inference_dense_layer_call_and_return_conditional_losses_36299

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesu
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis—
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis„
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/ConstА
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1И
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackХ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЙ
Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/transpose_1/permђ
Tensordot/transpose_1	Transpose Tensordot/ReadVariableOp:value:0#Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	А2
Tensordot/transpose_1З
Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"А      2
Tensordot/Reshape_1/shapeЮ
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0"Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2
Tensordot/Reshape_1Ъ
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Ф
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЛ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAdd†
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
Њ
©
(__inference_conv2d_1_layer_call_fn_36207

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_361992
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
н&
ш
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_36439

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2м
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3В
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3і
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype02!
conv2d_transpose/ReadVariableOpс
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
Reluї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
Є"
«
G__inference_sequential_1_layer_call_and_return_conditional_losses_36670
input_1*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_23
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_25
1conv2d_transpose_2_statefulpartitionedcall_args_15
1conv2d_transpose_2_statefulpartitionedcall_args_25
1conv2d_transpose_3_statefulpartitionedcall_args_15
1conv2d_transpose_3_statefulpartitionedcall_args_2
identityИҐ(conv2d_transpose/StatefulPartitionedCallҐ*conv2d_transpose_1/StatefulPartitionedCallҐ*conv2d_transpose_2/StatefulPartitionedCallҐ*conv2d_transpose_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallЃ
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_1&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€А**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_366262!
dense_1/StatefulPartitionedCallН
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_364392*
(conv2d_transpose/StatefulPartitionedCall†
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_364862,
*conv2d_transpose_1/StatefulPartitionedCallҐ
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:01conv2d_transpose_2_statefulpartitionedcall_args_11conv2d_transpose_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_365332,
*conv2d_transpose_2/StatefulPartitionedCallҐ
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:01conv2d_transpose_3_statefulpartitionedcall_args_11conv2d_transpose_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_365802,
*conv2d_transpose_3/StatefulPartitionedCallх
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
н&
ъ
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_36580

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐconv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2м
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2м
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3В
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2м
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3≥
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_transpose/ReadVariableOpс
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*
paddingVALID*
strides
2
conv2d_transposeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2
Reluї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:& "
 
_user_specified_nameinputs
р
Ѕ
,__inference_sequential_2_layer_call_fn_36955
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20* 
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_369322
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€`ј::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
ж
Џ
A__inference_conv2d_layer_call_and_return_conditional_losses_36178

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rateХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpґ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЪ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
”
х
G__inference_sequential_2_layer_call_and_return_conditional_losses_36826
input_1-
)sequential_statefulpartitionedcall_args_1-
)sequential_statefulpartitionedcall_args_2-
)sequential_statefulpartitionedcall_args_3-
)sequential_statefulpartitionedcall_args_4-
)sequential_statefulpartitionedcall_args_5-
)sequential_statefulpartitionedcall_args_6-
)sequential_statefulpartitionedcall_args_7-
)sequential_statefulpartitionedcall_args_8-
)sequential_statefulpartitionedcall_args_9.
*sequential_statefulpartitionedcall_args_10/
+sequential_1_statefulpartitionedcall_args_1/
+sequential_1_statefulpartitionedcall_args_2/
+sequential_1_statefulpartitionedcall_args_3/
+sequential_1_statefulpartitionedcall_args_4/
+sequential_1_statefulpartitionedcall_args_5/
+sequential_1_statefulpartitionedcall_args_6/
+sequential_1_statefulpartitionedcall_args_7/
+sequential_1_statefulpartitionedcall_args_8/
+sequential_1_statefulpartitionedcall_args_90
,sequential_1_statefulpartitionedcall_args_10
identityИҐ"sequential/StatefulPartitionedCallҐ$sequential_1/StatefulPartitionedCallЭ
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1)sequential_statefulpartitionedcall_args_1)sequential_statefulpartitionedcall_args_2)sequential_statefulpartitionedcall_args_3)sequential_statefulpartitionedcall_args_4)sequential_statefulpartitionedcall_args_5)sequential_statefulpartitionedcall_args_6)sequential_statefulpartitionedcall_args_7)sequential_statefulpartitionedcall_args_8)sequential_statefulpartitionedcall_args_9*sequential_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_363532$
"sequential/StatefulPartitionedCallн
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0+sequential_1_statefulpartitionedcall_args_1+sequential_1_statefulpartitionedcall_args_2+sequential_1_statefulpartitionedcall_args_3+sequential_1_statefulpartitionedcall_args_4+sequential_1_statefulpartitionedcall_args_5+sequential_1_statefulpartitionedcall_args_6+sequential_1_statefulpartitionedcall_args_7+sequential_1_statefulpartitionedcall_args_8+sequential_1_statefulpartitionedcall_args_9,sequential_1_statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_366922&
$sequential_1/StatefulPartitionedCallз
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€`ј::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
н
ј
,__inference_sequential_2_layer_call_fn_37470

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20
identityИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20* 
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_368812
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€`ј::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
е
«
E__inference_sequential_layer_call_and_return_conditional_losses_36312
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐdense/StatefulPartitionedCall®
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€0@**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_361782 
conv2d/StatefulPartitionedCall“
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_361992"
 conv2d_1/StatefulPartitionedCall‘
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_362202"
 conv2d_2/StatefulPartitionedCall’
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€А**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_362412"
 conv2d_3/StatefulPartitionedCall≈
dense/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_362992
dense/StatefulPartitionedCallђ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:€€€€€€€€€`ј::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
ущ
ж
G__inference_sequential_1_layer_call_and_return_conditional_losses_37819

inputs-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource?
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_3_biasadd_readvariableop_resource
identityИҐ'conv2d_transpose/BiasAdd/ReadVariableOpҐ0conv2d_transpose/conv2d_transpose/ReadVariableOpҐ)conv2d_transpose_1/BiasAdd/ReadVariableOpҐ2conv2d_transpose_1/conv2d_transpose/ReadVariableOpҐ)conv2d_transpose_2/BiasAdd/ReadVariableOpҐ2conv2d_transpose_2/conv2d_transpose/ReadVariableOpҐ)conv2d_transpose_3/BiasAdd/ReadVariableOpҐ2conv2d_transpose_3/conv2d_transpose/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐ dense_1/Tensordot/ReadVariableOpѓ
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axesЕ
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_1/Tensordot/freeh
dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_1/Tensordot/ShapeД
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axisщ
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2И
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis€
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2_1|
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const†
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/ProdА
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1®
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1А
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axisЎ
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concatђ
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stackђ
dense_1/Tensordot/transpose	Transposeinputs!dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
dense_1/Tensordot/transposeњ
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_1/Tensordot/ReshapeЩ
"dense_1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/Tensordot/transpose_1/permћ
dense_1/Tensordot/transpose_1	Transpose(dense_1/Tensordot/ReadVariableOp:value:0+dense_1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	А2
dense_1/Tensordot/transpose_1Ч
!dense_1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   А   2#
!dense_1/Tensordot/Reshape_1/shapeЊ
dense_1/Tensordot/Reshape_1Reshape!dense_1/Tensordot/transpose_1:y:0*dense_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2
dense_1/Tensordot/Reshape_1ї
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0$dense_1/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_1/Tensordot/MatMulБ
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
dense_1/Tensordot/Const_2Д
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axisе
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1µ
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dense_1/Tensordot•
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_1/BiasAdd/ReadVariableOpђ
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dense_1/BiasAddx
conv2d_transpose/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
conv2d_transpose/ShapeЦ
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stackЪ
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1Ъ
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2»
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_sliceЪ
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_1/stackЮ
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1Ю
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2“
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1Ъ
&conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_2/stackЮ
(conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_1Ю
(conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_2“
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_2/stack:output:01conv2d_transpose/strided_slice_2/stack_1:output:01conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_2r
conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul/y†
conv2d_transpose/mulMul)conv2d_transpose/strided_slice_1:output:0conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mulr
conv2d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose/add/yС
conv2d_transpose/addAddV2conv2d_transpose/mul:z:0conv2d_transpose/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/addv
conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul_1/y¶
conv2d_transpose/mul_1Mul)conv2d_transpose/strided_slice_2:output:0!conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mul_1v
conv2d_transpose/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose/add_1/yЩ
conv2d_transpose/add_1AddV2conv2d_transpose/mul_1:z:0!conv2d_transpose/add_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/add_1v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose/stack/3и
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0conv2d_transpose/add:z:0conv2d_transpose/add_1:z:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stackЪ
&conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_3/stackЮ
(conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_1Ю
(conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_2“
 conv2d_transpose/strided_slice_3StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_3/stack:output:01conv2d_transpose/strided_slice_3/stack_1:output:01conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_3з
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOpµ
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0dense_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2#
!conv2d_transpose/conv2d_transposeњ
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp÷
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_transpose/BiasAddУ
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_transpose/ReluЗ
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/ShapeЪ
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stackЮ
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1Ю
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2‘
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_sliceЮ
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_1/stackҐ
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1Ґ
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2ё
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1Ю
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_2/stackҐ
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_1Ґ
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_2ё
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_2v
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul/y®
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mulv
conv2d_transpose_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/add/yЩ
conv2d_transpose_1/addAddV2conv2d_transpose_1/mul:z:0!conv2d_transpose_1/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/addz
conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul_1/yЃ
conv2d_transpose_1/mul_1Mul+conv2d_transpose_1/strided_slice_2:output:0#conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mul_1z
conv2d_transpose_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/add_1/y°
conv2d_transpose_1/add_1AddV2conv2d_transpose_1/mul_1:z:0#conv2d_transpose_1/add_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/add_1z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_1/stack/3ф
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/add:z:0conv2d_transpose_1/add_1:z:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stackЮ
(conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_3/stackҐ
*conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_1Ґ
*conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_2ё
"conv2d_transpose_1/strided_slice_3StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_3/stack:output:03conv2d_transpose_1/strided_slice_3/stack_1:output:03conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_3м
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp»
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2%
#conv2d_transpose_1/conv2d_transpose≈
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOpё
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_transpose_1/BiasAddЩ
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_transpose_1/ReluЙ
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_2/ShapeЪ
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stackЮ
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1Ю
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2‘
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_sliceЮ
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice_1/stackҐ
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1Ґ
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2ё
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/Shape:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1Ю
(conv2d_transpose_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice_2/stackҐ
*conv2d_transpose_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_2/stack_1Ґ
*conv2d_transpose_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_2/stack_2ё
"conv2d_transpose_2/strided_slice_2StridedSlice!conv2d_transpose_2/Shape:output:01conv2d_transpose_2/strided_slice_2/stack:output:03conv2d_transpose_2/strided_slice_2/stack_1:output:03conv2d_transpose_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_2v
conv2d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/mul/y®
conv2d_transpose_2/mulMul+conv2d_transpose_2/strided_slice_1:output:0!conv2d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_2/mulv
conv2d_transpose_2/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_2/add/yЩ
conv2d_transpose_2/addAddV2conv2d_transpose_2/mul:z:0!conv2d_transpose_2/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_2/addz
conv2d_transpose_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/mul_1/yЃ
conv2d_transpose_2/mul_1Mul+conv2d_transpose_2/strided_slice_2:output:0#conv2d_transpose_2/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_2/mul_1z
conv2d_transpose_2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_2/add_1/y°
conv2d_transpose_2/add_1AddV2conv2d_transpose_2/mul_1:z:0#conv2d_transpose_2/add_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_2/add_1z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_2/stack/3ф
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0conv2d_transpose_2/add:z:0conv2d_transpose_2/add_1:z:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stackЮ
(conv2d_transpose_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_3/stackҐ
*conv2d_transpose_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_3/stack_1Ґ
*conv2d_transpose_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_3/stack_2ё
"conv2d_transpose_2/strided_slice_3StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_3/stack:output:03conv2d_transpose_2/strided_slice_3/stack_1:output:03conv2d_transpose_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_3м
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp 
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€0@*
paddingVALID*
strides
2%
#conv2d_transpose_2/conv2d_transpose≈
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOpё
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@2
conv2d_transpose_2/BiasAddЩ
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€0@2
conv2d_transpose_2/ReluЙ
conv2d_transpose_3/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_3/ShapeЪ
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stackЮ
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1Ю
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2‘
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_sliceЮ
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice_1/stackҐ
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1Ґ
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2ё
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/Shape:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1Ю
(conv2d_transpose_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice_2/stackҐ
*conv2d_transpose_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_2/stack_1Ґ
*conv2d_transpose_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_2/stack_2ё
"conv2d_transpose_3/strided_slice_2StridedSlice!conv2d_transpose_3/Shape:output:01conv2d_transpose_3/strided_slice_2/stack:output:03conv2d_transpose_3/strided_slice_2/stack_1:output:03conv2d_transpose_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_2v
conv2d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/mul/y®
conv2d_transpose_3/mulMul+conv2d_transpose_3/strided_slice_1:output:0!conv2d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_3/mulv
conv2d_transpose_3/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_3/add/yЩ
conv2d_transpose_3/addAddV2conv2d_transpose_3/mul:z:0!conv2d_transpose_3/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_3/addz
conv2d_transpose_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/mul_1/yЃ
conv2d_transpose_3/mul_1Mul+conv2d_transpose_3/strided_slice_2:output:0#conv2d_transpose_3/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_3/mul_1z
conv2d_transpose_3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_3/add_1/y°
conv2d_transpose_3/add_1AddV2conv2d_transpose_3/mul_1:z:0#conv2d_transpose_3/add_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_3/add_1z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/3ф
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0conv2d_transpose_3/add:z:0conv2d_transpose_3/add_1:z:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stackЮ
(conv2d_transpose_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_3/stackҐ
*conv2d_transpose_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_3/stack_1Ґ
*conv2d_transpose_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_3/stack_2ё
"conv2d_transpose_3/strided_slice_3StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_3/stack:output:03conv2d_transpose_3/strided_slice_3/stack_1:output:03conv2d_transpose_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_3м
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpЋ
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_2/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€`ј*
paddingVALID*
strides
2%
#conv2d_transpose_3/conv2d_transpose≈
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOpя
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`ј2
conv2d_transpose_3/BiasAddЪ
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€`ј2
conv2d_transpose_3/Relu∆
IdentityIdentity%conv2d_transpose_3/Relu:activations:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€`ј2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
Ї
І
&__inference_conv2d_layer_call_fn_36186

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИҐStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_361782
StatefulPartitionedCall®
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
е
«
E__inference_sequential_layer_call_and_return_conditional_losses_36331
input_1)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2+
'conv2d_3_statefulpartitionedcall_args_1+
'conv2d_3_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐdense/StatefulPartitionedCall®
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€0@**
config_proto

GPU 

CPU2J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_361782 
conv2d/StatefulPartitionedCall“
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_361992"
 conv2d_1/StatefulPartitionedCall‘
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_362202"
 conv2d_2/StatefulPartitionedCall’
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0'conv2d_3_statefulpartitionedcall_args_1'conv2d_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€А**
config_proto

GPU 

CPU2J 8*L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_362412"
 conv2d_3/StatefulPartitionedCall≈
dense/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€**
config_proto

GPU 

CPU2J 8*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_362992
dense/StatefulPartitionedCallђ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:€€€€€€€€€`ј::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
ущ
ж
G__inference_sequential_1_layer_call_and_return_conditional_losses_37989

inputs-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource?
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_3_biasadd_readvariableop_resource
identityИҐ'conv2d_transpose/BiasAdd/ReadVariableOpҐ0conv2d_transpose/conv2d_transpose/ReadVariableOpҐ)conv2d_transpose_1/BiasAdd/ReadVariableOpҐ2conv2d_transpose_1/conv2d_transpose/ReadVariableOpҐ)conv2d_transpose_2/BiasAdd/ReadVariableOpҐ2conv2d_transpose_2/conv2d_transpose/ReadVariableOpҐ)conv2d_transpose_3/BiasAdd/ReadVariableOpҐ2conv2d_transpose_3/conv2d_transpose/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐ dense_1/Tensordot/ReadVariableOpѓ
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axesЕ
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2
dense_1/Tensordot/freeh
dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_1/Tensordot/ShapeД
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axisщ
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2И
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis€
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2_1|
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const†
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/ProdА
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1®
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1А
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axisЎ
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concatђ
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stackђ
dense_1/Tensordot/transpose	Transposeinputs!dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
dense_1/Tensordot/transposeњ
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_1/Tensordot/ReshapeЩ
"dense_1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_1/Tensordot/transpose_1/permћ
dense_1/Tensordot/transpose_1	Transpose(dense_1/Tensordot/ReadVariableOp:value:0+dense_1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	А2
dense_1/Tensordot/transpose_1Ч
!dense_1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   А   2#
!dense_1/Tensordot/Reshape_1/shapeЊ
dense_1/Tensordot/Reshape_1Reshape!dense_1/Tensordot/transpose_1:y:0*dense_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А2
dense_1/Tensordot/Reshape_1ї
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0$dense_1/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_1/Tensordot/MatMulБ
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
dense_1/Tensordot/Const_2Д
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axisе
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1µ
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dense_1/Tensordot•
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_1/BiasAdd/ReadVariableOpђ
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2
dense_1/BiasAddx
conv2d_transpose/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
conv2d_transpose/ShapeЦ
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stackЪ
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1Ъ
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2»
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_sliceЪ
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_1/stackЮ
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1Ю
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2“
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1Ъ
&conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice_2/stackЮ
(conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_1Ю
(conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_2/stack_2“
 conv2d_transpose/strided_slice_2StridedSliceconv2d_transpose/Shape:output:0/conv2d_transpose/strided_slice_2/stack:output:01conv2d_transpose/strided_slice_2/stack_1:output:01conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_2r
conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul/y†
conv2d_transpose/mulMul)conv2d_transpose/strided_slice_1:output:0conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mulr
conv2d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose/add/yС
conv2d_transpose/addAddV2conv2d_transpose/mul:z:0conv2d_transpose/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/addv
conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/mul_1/y¶
conv2d_transpose/mul_1Mul)conv2d_transpose/strided_slice_2:output:0!conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/mul_1v
conv2d_transpose/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose/add_1/yЩ
conv2d_transpose/add_1AddV2conv2d_transpose/mul_1:z:0!conv2d_transpose/add_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose/add_1v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose/stack/3и
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0conv2d_transpose/add:z:0conv2d_transpose/add_1:z:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stackЪ
&conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_3/stackЮ
(conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_1Ю
(conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_3/stack_2“
 conv2d_transpose/strided_slice_3StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_3/stack:output:01conv2d_transpose/strided_slice_3/stack_1:output:01conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_3з
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOpµ
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0dense_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2#
!conv2d_transpose/conv2d_transposeњ
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp÷
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_transpose/BiasAddУ
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_transpose/ReluЗ
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/ShapeЪ
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stackЮ
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1Ю
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2‘
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_sliceЮ
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_1/stackҐ
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1Ґ
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2ё
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1Ю
(conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice_2/stackҐ
*conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_1Ґ
*conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_2/stack_2ё
"conv2d_transpose_1/strided_slice_2StridedSlice!conv2d_transpose_1/Shape:output:01conv2d_transpose_1/strided_slice_2/stack:output:03conv2d_transpose_1/strided_slice_2/stack_1:output:03conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_2v
conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul/y®
conv2d_transpose_1/mulMul+conv2d_transpose_1/strided_slice_1:output:0!conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mulv
conv2d_transpose_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/add/yЩ
conv2d_transpose_1/addAddV2conv2d_transpose_1/mul:z:0!conv2d_transpose_1/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/addz
conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/mul_1/yЃ
conv2d_transpose_1/mul_1Mul+conv2d_transpose_1/strided_slice_2:output:0#conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/mul_1z
conv2d_transpose_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/add_1/y°
conv2d_transpose_1/add_1AddV2conv2d_transpose_1/mul_1:z:0#conv2d_transpose_1/add_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_1/add_1z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_1/stack/3ф
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0conv2d_transpose_1/add:z:0conv2d_transpose_1/add_1:z:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stackЮ
(conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_3/stackҐ
*conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_1Ґ
*conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_3/stack_2ё
"conv2d_transpose_1/strided_slice_3StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_3/stack:output:03conv2d_transpose_1/strided_slice_3/stack_1:output:03conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_3м
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp»
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2%
#conv2d_transpose_1/conv2d_transpose≈
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOpё
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_transpose_1/BiasAddЩ
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2
conv2d_transpose_1/ReluЙ
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_2/ShapeЪ
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stackЮ
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1Ю
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2‘
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_sliceЮ
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice_1/stackҐ
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1Ґ
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2ё
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/Shape:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1Ю
(conv2d_transpose_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice_2/stackҐ
*conv2d_transpose_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_2/stack_1Ґ
*conv2d_transpose_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_2/stack_2ё
"conv2d_transpose_2/strided_slice_2StridedSlice!conv2d_transpose_2/Shape:output:01conv2d_transpose_2/strided_slice_2/stack:output:03conv2d_transpose_2/strided_slice_2/stack_1:output:03conv2d_transpose_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_2v
conv2d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/mul/y®
conv2d_transpose_2/mulMul+conv2d_transpose_2/strided_slice_1:output:0!conv2d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_2/mulv
conv2d_transpose_2/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_2/add/yЩ
conv2d_transpose_2/addAddV2conv2d_transpose_2/mul:z:0!conv2d_transpose_2/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_2/addz
conv2d_transpose_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/mul_1/yЃ
conv2d_transpose_2/mul_1Mul+conv2d_transpose_2/strided_slice_2:output:0#conv2d_transpose_2/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_2/mul_1z
conv2d_transpose_2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_2/add_1/y°
conv2d_transpose_2/add_1AddV2conv2d_transpose_2/mul_1:z:0#conv2d_transpose_2/add_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_2/add_1z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_2/stack/3ф
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0conv2d_transpose_2/add:z:0conv2d_transpose_2/add_1:z:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stackЮ
(conv2d_transpose_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_3/stackҐ
*conv2d_transpose_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_3/stack_1Ґ
*conv2d_transpose_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_3/stack_2ё
"conv2d_transpose_2/strided_slice_3StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_3/stack:output:03conv2d_transpose_2/strided_slice_3/stack_1:output:03conv2d_transpose_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_3м
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp 
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€0@*
paddingVALID*
strides
2%
#conv2d_transpose_2/conv2d_transpose≈
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOpё
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@2
conv2d_transpose_2/BiasAddЩ
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€0@2
conv2d_transpose_2/ReluЙ
conv2d_transpose_3/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_3/ShapeЪ
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stackЮ
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1Ю
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2‘
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_sliceЮ
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice_1/stackҐ
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1Ґ
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2ё
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/Shape:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1Ю
(conv2d_transpose_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice_2/stackҐ
*conv2d_transpose_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_2/stack_1Ґ
*conv2d_transpose_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_2/stack_2ё
"conv2d_transpose_3/strided_slice_2StridedSlice!conv2d_transpose_3/Shape:output:01conv2d_transpose_3/strided_slice_2/stack:output:03conv2d_transpose_3/strided_slice_2/stack_1:output:03conv2d_transpose_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_2v
conv2d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/mul/y®
conv2d_transpose_3/mulMul+conv2d_transpose_3/strided_slice_1:output:0!conv2d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_3/mulv
conv2d_transpose_3/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_3/add/yЩ
conv2d_transpose_3/addAddV2conv2d_transpose_3/mul:z:0!conv2d_transpose_3/add/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_3/addz
conv2d_transpose_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/mul_1/yЃ
conv2d_transpose_3/mul_1Mul+conv2d_transpose_3/strided_slice_2:output:0#conv2d_transpose_3/mul_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_3/mul_1z
conv2d_transpose_3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_3/add_1/y°
conv2d_transpose_3/add_1AddV2conv2d_transpose_3/mul_1:z:0#conv2d_transpose_3/add_1/y:output:0*
T0*
_output_shapes
: 2
conv2d_transpose_3/add_1z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/3ф
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0conv2d_transpose_3/add:z:0conv2d_transpose_3/add_1:z:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stackЮ
(conv2d_transpose_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_3/stackҐ
*conv2d_transpose_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_3/stack_1Ґ
*conv2d_transpose_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_3/stack_2ё
"conv2d_transpose_3/strided_slice_3StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_3/stack:output:03conv2d_transpose_3/strided_slice_3/stack_1:output:03conv2d_transpose_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_3м
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpЋ
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_2/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€`ј*
paddingVALID*
strides
2%
#conv2d_transpose_3/conv2d_transpose≈
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOpя
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`ј2
conv2d_transpose_3/BiasAddЪ
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€`ј2
conv2d_transpose_3/Relu∆
IdentityIdentity%conv2d_transpose_3/Relu:activations:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€`ј2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp:& "
 
_user_specified_nameinputs
µ"
∆
G__inference_sequential_1_layer_call_and_return_conditional_losses_36692

inputs*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_23
/conv2d_transpose_statefulpartitionedcall_args_13
/conv2d_transpose_statefulpartitionedcall_args_25
1conv2d_transpose_1_statefulpartitionedcall_args_15
1conv2d_transpose_1_statefulpartitionedcall_args_25
1conv2d_transpose_2_statefulpartitionedcall_args_15
1conv2d_transpose_2_statefulpartitionedcall_args_25
1conv2d_transpose_3_statefulpartitionedcall_args_15
1conv2d_transpose_3_statefulpartitionedcall_args_2
identityИҐ(conv2d_transpose/StatefulPartitionedCallҐ*conv2d_transpose_1/StatefulPartitionedCallҐ*conv2d_transpose_2/StatefulPartitionedCallҐ*conv2d_transpose_3/StatefulPartitionedCallҐdense_1/StatefulPartitionedCall≠
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputs&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*0
_output_shapes
:€€€€€€€€€А**
config_proto

GPU 

CPU2J 8*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_366262!
dense_1/StatefulPartitionedCallН
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0/conv2d_transpose_statefulpartitionedcall_args_1/conv2d_transpose_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_364392*
(conv2d_transpose/StatefulPartitionedCall†
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:01conv2d_transpose_1_statefulpartitionedcall_args_11conv2d_transpose_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_364862,
*conv2d_transpose_1/StatefulPartitionedCallҐ
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:01conv2d_transpose_2_statefulpartitionedcall_args_11conv2d_transpose_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_365332,
*conv2d_transpose_2/StatefulPartitionedCallҐ
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:01conv2d_transpose_3_statefulpartitionedcall_args_11conv2d_transpose_3_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€**
config_proto

GPU 

CPU2J 8*V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_365802,
*conv2d_transpose_3/StatefulPartitionedCallх
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
Џ
Ќ
*__inference_sequential_layer_call_fn_36400
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:€€€€€€€€€**
config_proto

GPU 

CPU2J 8*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_363872
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*W
_input_shapesF
D:€€€€€€€€€`ј::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
‘№
∞
 __inference__wrapped_model_36165
input_1A
=sequential_2_sequential_conv2d_conv2d_readvariableop_resourceB
>sequential_2_sequential_conv2d_biasadd_readvariableop_resourceC
?sequential_2_sequential_conv2d_1_conv2d_readvariableop_resourceD
@sequential_2_sequential_conv2d_1_biasadd_readvariableop_resourceC
?sequential_2_sequential_conv2d_2_conv2d_readvariableop_resourceD
@sequential_2_sequential_conv2d_2_biasadd_readvariableop_resourceC
?sequential_2_sequential_conv2d_3_conv2d_readvariableop_resourceD
@sequential_2_sequential_conv2d_3_biasadd_readvariableop_resourceC
?sequential_2_sequential_dense_tensordot_readvariableop_resourceA
=sequential_2_sequential_dense_biasadd_readvariableop_resourceG
Csequential_2_sequential_1_dense_1_tensordot_readvariableop_resourceE
Asequential_2_sequential_1_dense_1_biasadd_readvariableop_resourceW
Ssequential_2_sequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resourceN
Jsequential_2_sequential_1_conv2d_transpose_biasadd_readvariableop_resourceY
Usequential_2_sequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceP
Lsequential_2_sequential_1_conv2d_transpose_1_biasadd_readvariableop_resourceY
Usequential_2_sequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceP
Lsequential_2_sequential_1_conv2d_transpose_2_biasadd_readvariableop_resourceY
Usequential_2_sequential_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resourceP
Lsequential_2_sequential_1_conv2d_transpose_3_biasadd_readvariableop_resource
identityИҐ5sequential_2/sequential/conv2d/BiasAdd/ReadVariableOpҐ4sequential_2/sequential/conv2d/Conv2D/ReadVariableOpҐ7sequential_2/sequential/conv2d_1/BiasAdd/ReadVariableOpҐ6sequential_2/sequential/conv2d_1/Conv2D/ReadVariableOpҐ7sequential_2/sequential/conv2d_2/BiasAdd/ReadVariableOpҐ6sequential_2/sequential/conv2d_2/Conv2D/ReadVariableOpҐ7sequential_2/sequential/conv2d_3/BiasAdd/ReadVariableOpҐ6sequential_2/sequential/conv2d_3/Conv2D/ReadVariableOpҐ4sequential_2/sequential/dense/BiasAdd/ReadVariableOpҐ6sequential_2/sequential/dense/Tensordot/ReadVariableOpҐAsequential_2/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpҐJsequential_2/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpҐCsequential_2/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpҐLsequential_2/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpҐCsequential_2/sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpҐLsequential_2/sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpҐCsequential_2/sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOpҐLsequential_2/sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpҐ8sequential_2/sequential_1/dense_1/BiasAdd/ReadVariableOpҐ:sequential_2/sequential_1/dense_1/Tensordot/ReadVariableOpт
4sequential_2/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp=sequential_2_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype026
4sequential_2/sequential/conv2d/Conv2D/ReadVariableOpВ
%sequential_2/sequential/conv2d/Conv2DConv2Dinput_1<sequential_2/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@*
paddingVALID*
strides
2'
%sequential_2/sequential/conv2d/Conv2Dй
5sequential_2/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp>sequential_2_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5sequential_2/sequential/conv2d/BiasAdd/ReadVariableOpД
&sequential_2/sequential/conv2d/BiasAddBiasAdd.sequential_2/sequential/conv2d/Conv2D:output:0=sequential_2/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@2(
&sequential_2/sequential/conv2d/BiasAddљ
#sequential_2/sequential/conv2d/ReluRelu/sequential_2/sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€0@2%
#sequential_2/sequential/conv2d/Reluш
6sequential_2/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp?sequential_2_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype028
6sequential_2/sequential/conv2d_1/Conv2D/ReadVariableOp≤
'sequential_2/sequential/conv2d_1/Conv2DConv2D1sequential_2/sequential/conv2d/Relu:activations:0>sequential_2/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2)
'sequential_2/sequential/conv2d_1/Conv2Dп
7sequential_2/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp@sequential_2_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7sequential_2/sequential/conv2d_1/BiasAdd/ReadVariableOpМ
(sequential_2/sequential/conv2d_1/BiasAddBiasAdd0sequential_2/sequential/conv2d_1/Conv2D:output:0?sequential_2/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2*
(sequential_2/sequential/conv2d_1/BiasAdd√
%sequential_2/sequential/conv2d_1/ReluRelu1sequential_2/sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2'
%sequential_2/sequential/conv2d_1/Reluш
6sequential_2/sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp?sequential_2_sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype028
6sequential_2/sequential/conv2d_2/Conv2D/ReadVariableOpі
'sequential_2/sequential/conv2d_2/Conv2DConv2D3sequential_2/sequential/conv2d_1/Relu:activations:0>sequential_2/sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2)
'sequential_2/sequential/conv2d_2/Conv2Dп
7sequential_2/sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp@sequential_2_sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7sequential_2/sequential/conv2d_2/BiasAdd/ReadVariableOpМ
(sequential_2/sequential/conv2d_2/BiasAddBiasAdd0sequential_2/sequential/conv2d_2/Conv2D:output:0?sequential_2/sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@2*
(sequential_2/sequential/conv2d_2/BiasAdd√
%sequential_2/sequential/conv2d_2/ReluRelu1sequential_2/sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@2'
%sequential_2/sequential/conv2d_2/Reluщ
6sequential_2/sequential/conv2d_3/Conv2D/ReadVariableOpReadVariableOp?sequential_2_sequential_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype028
6sequential_2/sequential/conv2d_3/Conv2D/ReadVariableOpµ
'sequential_2/sequential/conv2d_3/Conv2DConv2D3sequential_2/sequential/conv2d_2/Relu:activations:0>sequential_2/sequential/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingVALID*
strides
2)
'sequential_2/sequential/conv2d_3/Conv2Dр
7sequential_2/sequential/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp@sequential_2_sequential_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype029
7sequential_2/sequential/conv2d_3/BiasAdd/ReadVariableOpН
(sequential_2/sequential/conv2d_3/BiasAddBiasAdd0sequential_2/sequential/conv2d_3/Conv2D:output:0?sequential_2/sequential/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2*
(sequential_2/sequential/conv2d_3/BiasAddƒ
%sequential_2/sequential/conv2d_3/ReluRelu1sequential_2/sequential/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2'
%sequential_2/sequential/conv2d_3/Reluс
6sequential_2/sequential/dense/Tensordot/ReadVariableOpReadVariableOp?sequential_2_sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype028
6sequential_2/sequential/dense/Tensordot/ReadVariableOp¶
,sequential_2/sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_2/sequential/dense/Tensordot/axes±
,sequential_2/sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          2.
,sequential_2/sequential/dense/Tensordot/freeЅ
-sequential_2/sequential/dense/Tensordot/ShapeShape3sequential_2/sequential/conv2d_3/Relu:activations:0*
T0*
_output_shapes
:2/
-sequential_2/sequential/dense/Tensordot/Shape∞
5sequential_2/sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5sequential_2/sequential/dense/Tensordot/GatherV2/axisз
0sequential_2/sequential/dense/Tensordot/GatherV2GatherV26sequential_2/sequential/dense/Tensordot/Shape:output:05sequential_2/sequential/dense/Tensordot/free:output:0>sequential_2/sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:22
0sequential_2/sequential/dense/Tensordot/GatherV2і
7sequential_2/sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7sequential_2/sequential/dense/Tensordot/GatherV2_1/axisн
2sequential_2/sequential/dense/Tensordot/GatherV2_1GatherV26sequential_2/sequential/dense/Tensordot/Shape:output:05sequential_2/sequential/dense/Tensordot/axes:output:0@sequential_2/sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2sequential_2/sequential/dense/Tensordot/GatherV2_1®
-sequential_2/sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_2/sequential/dense/Tensordot/Constш
,sequential_2/sequential/dense/Tensordot/ProdProd9sequential_2/sequential/dense/Tensordot/GatherV2:output:06sequential_2/sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2.
,sequential_2/sequential/dense/Tensordot/Prodђ
/sequential_2/sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/sequential_2/sequential/dense/Tensordot/Const_1А
.sequential_2/sequential/dense/Tensordot/Prod_1Prod;sequential_2/sequential/dense/Tensordot/GatherV2_1:output:08sequential_2/sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 20
.sequential_2/sequential/dense/Tensordot/Prod_1ђ
3sequential_2/sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3sequential_2/sequential/dense/Tensordot/concat/axis∆
.sequential_2/sequential/dense/Tensordot/concatConcatV25sequential_2/sequential/dense/Tensordot/free:output:05sequential_2/sequential/dense/Tensordot/axes:output:0<sequential_2/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:20
.sequential_2/sequential/dense/Tensordot/concatД
-sequential_2/sequential/dense/Tensordot/stackPack5sequential_2/sequential/dense/Tensordot/Prod:output:07sequential_2/sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2/
-sequential_2/sequential/dense/Tensordot/stackЬ
1sequential_2/sequential/dense/Tensordot/transpose	Transpose3sequential_2/sequential/conv2d_3/Relu:activations:07sequential_2/sequential/dense/Tensordot/concat:output:0*
T0*0
_output_shapes
:€€€€€€€€€А23
1sequential_2/sequential/dense/Tensordot/transposeЧ
/sequential_2/sequential/dense/Tensordot/ReshapeReshape5sequential_2/sequential/dense/Tensordot/transpose:y:06sequential_2/sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€21
/sequential_2/sequential/dense/Tensordot/Reshape≈
8sequential_2/sequential/dense/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2:
8sequential_2/sequential/dense/Tensordot/transpose_1/perm§
3sequential_2/sequential/dense/Tensordot/transpose_1	Transpose>sequential_2/sequential/dense/Tensordot/ReadVariableOp:value:0Asequential_2/sequential/dense/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	А25
3sequential_2/sequential/dense/Tensordot/transpose_1√
7sequential_2/sequential/dense/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"А      29
7sequential_2/sequential/dense/Tensordot/Reshape_1/shapeЦ
1sequential_2/sequential/dense/Tensordot/Reshape_1Reshape7sequential_2/sequential/dense/Tensordot/transpose_1:y:0@sequential_2/sequential/dense/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А23
1sequential_2/sequential/dense/Tensordot/Reshape_1Т
.sequential_2/sequential/dense/Tensordot/MatMulMatMul8sequential_2/sequential/dense/Tensordot/Reshape:output:0:sequential_2/sequential/dense/Tensordot/Reshape_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€20
.sequential_2/sequential/dense/Tensordot/MatMulђ
/sequential_2/sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential_2/sequential/dense/Tensordot/Const_2∞
5sequential_2/sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5sequential_2/sequential/dense/Tensordot/concat_1/axis”
0sequential_2/sequential/dense/Tensordot/concat_1ConcatV29sequential_2/sequential/dense/Tensordot/GatherV2:output:08sequential_2/sequential/dense/Tensordot/Const_2:output:0>sequential_2/sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:22
0sequential_2/sequential/dense/Tensordot/concat_1М
'sequential_2/sequential/dense/TensordotReshape8sequential_2/sequential/dense/Tensordot/MatMul:product:09sequential_2/sequential/dense/Tensordot/concat_1:output:0*
T0*/
_output_shapes
:€€€€€€€€€2)
'sequential_2/sequential/dense/Tensordotж
4sequential_2/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp=sequential_2_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4sequential_2/sequential/dense/BiasAdd/ReadVariableOpГ
%sequential_2/sequential/dense/BiasAddBiasAdd0sequential_2/sequential/dense/Tensordot:output:0<sequential_2/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2'
%sequential_2/sequential/dense/BiasAddэ
:sequential_2/sequential_1/dense_1/Tensordot/ReadVariableOpReadVariableOpCsequential_2_sequential_1_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	А*
dtype02<
:sequential_2/sequential_1/dense_1/Tensordot/ReadVariableOpЃ
0sequential_2/sequential_1/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0sequential_2/sequential_1/dense_1/Tensordot/axesє
0sequential_2/sequential_1/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          22
0sequential_2/sequential_1/dense_1/Tensordot/freeƒ
1sequential_2/sequential_1/dense_1/Tensordot/ShapeShape.sequential_2/sequential/dense/BiasAdd:output:0*
T0*
_output_shapes
:23
1sequential_2/sequential_1/dense_1/Tensordot/ShapeЄ
9sequential_2/sequential_1/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9sequential_2/sequential_1/dense_1/Tensordot/GatherV2/axisы
4sequential_2/sequential_1/dense_1/Tensordot/GatherV2GatherV2:sequential_2/sequential_1/dense_1/Tensordot/Shape:output:09sequential_2/sequential_1/dense_1/Tensordot/free:output:0Bsequential_2/sequential_1/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4sequential_2/sequential_1/dense_1/Tensordot/GatherV2Љ
;sequential_2/sequential_1/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;sequential_2/sequential_1/dense_1/Tensordot/GatherV2_1/axisБ
6sequential_2/sequential_1/dense_1/Tensordot/GatherV2_1GatherV2:sequential_2/sequential_1/dense_1/Tensordot/Shape:output:09sequential_2/sequential_1/dense_1/Tensordot/axes:output:0Dsequential_2/sequential_1/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6sequential_2/sequential_1/dense_1/Tensordot/GatherV2_1∞
1sequential_2/sequential_1/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1sequential_2/sequential_1/dense_1/Tensordot/ConstИ
0sequential_2/sequential_1/dense_1/Tensordot/ProdProd=sequential_2/sequential_1/dense_1/Tensordot/GatherV2:output:0:sequential_2/sequential_1/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0sequential_2/sequential_1/dense_1/Tensordot/Prodі
3sequential_2/sequential_1/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_2/sequential_1/dense_1/Tensordot/Const_1Р
2sequential_2/sequential_1/dense_1/Tensordot/Prod_1Prod?sequential_2/sequential_1/dense_1/Tensordot/GatherV2_1:output:0<sequential_2/sequential_1/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2sequential_2/sequential_1/dense_1/Tensordot/Prod_1і
7sequential_2/sequential_1/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7sequential_2/sequential_1/dense_1/Tensordot/concat/axisЏ
2sequential_2/sequential_1/dense_1/Tensordot/concatConcatV29sequential_2/sequential_1/dense_1/Tensordot/free:output:09sequential_2/sequential_1/dense_1/Tensordot/axes:output:0@sequential_2/sequential_1/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2sequential_2/sequential_1/dense_1/Tensordot/concatФ
1sequential_2/sequential_1/dense_1/Tensordot/stackPack9sequential_2/sequential_1/dense_1/Tensordot/Prod:output:0;sequential_2/sequential_1/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1sequential_2/sequential_1/dense_1/Tensordot/stackҐ
5sequential_2/sequential_1/dense_1/Tensordot/transpose	Transpose.sequential_2/sequential/dense/BiasAdd:output:0;sequential_2/sequential_1/dense_1/Tensordot/concat:output:0*
T0*/
_output_shapes
:€€€€€€€€€27
5sequential_2/sequential_1/dense_1/Tensordot/transposeІ
3sequential_2/sequential_1/dense_1/Tensordot/ReshapeReshape9sequential_2/sequential_1/dense_1/Tensordot/transpose:y:0:sequential_2/sequential_1/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€25
3sequential_2/sequential_1/dense_1/Tensordot/ReshapeЌ
<sequential_2/sequential_1/dense_1/Tensordot/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2>
<sequential_2/sequential_1/dense_1/Tensordot/transpose_1/permі
7sequential_2/sequential_1/dense_1/Tensordot/transpose_1	TransposeBsequential_2/sequential_1/dense_1/Tensordot/ReadVariableOp:value:0Esequential_2/sequential_1/dense_1/Tensordot/transpose_1/perm:output:0*
T0*
_output_shapes
:	А29
7sequential_2/sequential_1/dense_1/Tensordot/transpose_1Ћ
;sequential_2/sequential_1/dense_1/Tensordot/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   А   2=
;sequential_2/sequential_1/dense_1/Tensordot/Reshape_1/shape¶
5sequential_2/sequential_1/dense_1/Tensordot/Reshape_1Reshape;sequential_2/sequential_1/dense_1/Tensordot/transpose_1:y:0Dsequential_2/sequential_1/dense_1/Tensordot/Reshape_1/shape:output:0*
T0*
_output_shapes
:	А27
5sequential_2/sequential_1/dense_1/Tensordot/Reshape_1£
2sequential_2/sequential_1/dense_1/Tensordot/MatMulMatMul<sequential_2/sequential_1/dense_1/Tensordot/Reshape:output:0>sequential_2/sequential_1/dense_1/Tensordot/Reshape_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€А24
2sequential_2/sequential_1/dense_1/Tensordot/MatMulµ
3sequential_2/sequential_1/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А25
3sequential_2/sequential_1/dense_1/Tensordot/Const_2Є
9sequential_2/sequential_1/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9sequential_2/sequential_1/dense_1/Tensordot/concat_1/axisз
4sequential_2/sequential_1/dense_1/Tensordot/concat_1ConcatV2=sequential_2/sequential_1/dense_1/Tensordot/GatherV2:output:0<sequential_2/sequential_1/dense_1/Tensordot/Const_2:output:0Bsequential_2/sequential_1/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4sequential_2/sequential_1/dense_1/Tensordot/concat_1Э
+sequential_2/sequential_1/dense_1/TensordotReshape<sequential_2/sequential_1/dense_1/Tensordot/MatMul:product:0=sequential_2/sequential_1/dense_1/Tensordot/concat_1:output:0*
T0*0
_output_shapes
:€€€€€€€€€А2-
+sequential_2/sequential_1/dense_1/Tensordotу
8sequential_2/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOpAsequential_2_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02:
8sequential_2/sequential_1/dense_1/BiasAdd/ReadVariableOpФ
)sequential_2/sequential_1/dense_1/BiasAddBiasAdd4sequential_2/sequential_1/dense_1/Tensordot:output:0@sequential_2/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А2+
)sequential_2/sequential_1/dense_1/BiasAdd∆
0sequential_2/sequential_1/conv2d_transpose/ShapeShape2sequential_2/sequential_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:22
0sequential_2/sequential_1/conv2d_transpose/Shape 
>sequential_2/sequential_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>sequential_2/sequential_1/conv2d_transpose/strided_slice/stackќ
@sequential_2/sequential_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_2/sequential_1/conv2d_transpose/strided_slice/stack_1ќ
@sequential_2/sequential_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_2/sequential_1/conv2d_transpose/strided_slice/stack_2д
8sequential_2/sequential_1/conv2d_transpose/strided_sliceStridedSlice9sequential_2/sequential_1/conv2d_transpose/Shape:output:0Gsequential_2/sequential_1/conv2d_transpose/strided_slice/stack:output:0Isequential_2/sequential_1/conv2d_transpose/strided_slice/stack_1:output:0Isequential_2/sequential_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8sequential_2/sequential_1/conv2d_transpose/strided_sliceќ
@sequential_2/sequential_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_2/sequential_1/conv2d_transpose/strided_slice_1/stack“
Bsequential_2/sequential_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose/strided_slice_1/stack_1“
Bsequential_2/sequential_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose/strided_slice_1/stack_2о
:sequential_2/sequential_1/conv2d_transpose/strided_slice_1StridedSlice9sequential_2/sequential_1/conv2d_transpose/Shape:output:0Isequential_2/sequential_1/conv2d_transpose/strided_slice_1/stack:output:0Ksequential_2/sequential_1/conv2d_transpose/strided_slice_1/stack_1:output:0Ksequential_2/sequential_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_2/sequential_1/conv2d_transpose/strided_slice_1ќ
@sequential_2/sequential_1/conv2d_transpose/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_2/sequential_1/conv2d_transpose/strided_slice_2/stack“
Bsequential_2/sequential_1/conv2d_transpose/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose/strided_slice_2/stack_1“
Bsequential_2/sequential_1/conv2d_transpose/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose/strided_slice_2/stack_2о
:sequential_2/sequential_1/conv2d_transpose/strided_slice_2StridedSlice9sequential_2/sequential_1/conv2d_transpose/Shape:output:0Isequential_2/sequential_1/conv2d_transpose/strided_slice_2/stack:output:0Ksequential_2/sequential_1/conv2d_transpose/strided_slice_2/stack_1:output:0Ksequential_2/sequential_1/conv2d_transpose/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_2/sequential_1/conv2d_transpose/strided_slice_2¶
0sequential_2/sequential_1/conv2d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :22
0sequential_2/sequential_1/conv2d_transpose/mul/yИ
.sequential_2/sequential_1/conv2d_transpose/mulMulCsequential_2/sequential_1/conv2d_transpose/strided_slice_1:output:09sequential_2/sequential_1/conv2d_transpose/mul/y:output:0*
T0*
_output_shapes
: 20
.sequential_2/sequential_1/conv2d_transpose/mul¶
0sequential_2/sequential_1/conv2d_transpose/add/yConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_2/sequential_1/conv2d_transpose/add/yщ
.sequential_2/sequential_1/conv2d_transpose/addAddV22sequential_2/sequential_1/conv2d_transpose/mul:z:09sequential_2/sequential_1/conv2d_transpose/add/y:output:0*
T0*
_output_shapes
: 20
.sequential_2/sequential_1/conv2d_transpose/add™
2sequential_2/sequential_1/conv2d_transpose/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_2/sequential_1/conv2d_transpose/mul_1/yО
0sequential_2/sequential_1/conv2d_transpose/mul_1MulCsequential_2/sequential_1/conv2d_transpose/strided_slice_2:output:0;sequential_2/sequential_1/conv2d_transpose/mul_1/y:output:0*
T0*
_output_shapes
: 22
0sequential_2/sequential_1/conv2d_transpose/mul_1™
2sequential_2/sequential_1/conv2d_transpose/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 24
2sequential_2/sequential_1/conv2d_transpose/add_1/yБ
0sequential_2/sequential_1/conv2d_transpose/add_1AddV24sequential_2/sequential_1/conv2d_transpose/mul_1:z:0;sequential_2/sequential_1/conv2d_transpose/add_1/y:output:0*
T0*
_output_shapes
: 22
0sequential_2/sequential_1/conv2d_transpose/add_1™
2sequential_2/sequential_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@24
2sequential_2/sequential_1/conv2d_transpose/stack/3Д
0sequential_2/sequential_1/conv2d_transpose/stackPackAsequential_2/sequential_1/conv2d_transpose/strided_slice:output:02sequential_2/sequential_1/conv2d_transpose/add:z:04sequential_2/sequential_1/conv2d_transpose/add_1:z:0;sequential_2/sequential_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:22
0sequential_2/sequential_1/conv2d_transpose/stackќ
@sequential_2/sequential_1/conv2d_transpose/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_2/sequential_1/conv2d_transpose/strided_slice_3/stack“
Bsequential_2/sequential_1/conv2d_transpose/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose/strided_slice_3/stack_1“
Bsequential_2/sequential_1/conv2d_transpose/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose/strided_slice_3/stack_2о
:sequential_2/sequential_1/conv2d_transpose/strided_slice_3StridedSlice9sequential_2/sequential_1/conv2d_transpose/stack:output:0Isequential_2/sequential_1/conv2d_transpose/strided_slice_3/stack:output:0Ksequential_2/sequential_1/conv2d_transpose/strided_slice_3/stack_1:output:0Ksequential_2/sequential_1/conv2d_transpose/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_2/sequential_1/conv2d_transpose/strided_slice_3µ
Jsequential_2/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpSsequential_2_sequential_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@А*
dtype02L
Jsequential_2/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpЈ
;sequential_2/sequential_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput9sequential_2/sequential_1/conv2d_transpose/stack:output:0Rsequential_2/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:02sequential_2/sequential_1/dense_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2=
;sequential_2/sequential_1/conv2d_transpose/conv2d_transposeН
Asequential_2/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpJsequential_2_sequential_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02C
Asequential_2/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpЊ
2sequential_2/sequential_1/conv2d_transpose/BiasAddBiasAddDsequential_2/sequential_1/conv2d_transpose/conv2d_transpose:output:0Isequential_2/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@24
2sequential_2/sequential_1/conv2d_transpose/BiasAddб
/sequential_2/sequential_1/conv2d_transpose/ReluRelu;sequential_2/sequential_1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@21
/sequential_2/sequential_1/conv2d_transpose/Relu’
2sequential_2/sequential_1/conv2d_transpose_1/ShapeShape=sequential_2/sequential_1/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:24
2sequential_2/sequential_1/conv2d_transpose_1/Shapeќ
@sequential_2/sequential_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_2/sequential_1/conv2d_transpose_1/strided_slice/stack“
Bsequential_2/sequential_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose_1/strided_slice/stack_1“
Bsequential_2/sequential_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose_1/strided_slice/stack_2р
:sequential_2/sequential_1/conv2d_transpose_1/strided_sliceStridedSlice;sequential_2/sequential_1/conv2d_transpose_1/Shape:output:0Isequential_2/sequential_1/conv2d_transpose_1/strided_slice/stack:output:0Ksequential_2/sequential_1/conv2d_transpose_1/strided_slice/stack_1:output:0Ksequential_2/sequential_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_2/sequential_1/conv2d_transpose_1/strided_slice“
Bsequential_2/sequential_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose_1/strided_slice_1/stack÷
Dsequential_2/sequential_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_1/strided_slice_1/stack_1÷
Dsequential_2/sequential_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_1/strided_slice_1/stack_2ъ
<sequential_2/sequential_1/conv2d_transpose_1/strided_slice_1StridedSlice;sequential_2/sequential_1/conv2d_transpose_1/Shape:output:0Ksequential_2/sequential_1/conv2d_transpose_1/strided_slice_1/stack:output:0Msequential_2/sequential_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0Msequential_2/sequential_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_2/sequential_1/conv2d_transpose_1/strided_slice_1“
Bsequential_2/sequential_1/conv2d_transpose_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose_1/strided_slice_2/stack÷
Dsequential_2/sequential_1/conv2d_transpose_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_1/strided_slice_2/stack_1÷
Dsequential_2/sequential_1/conv2d_transpose_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_1/strided_slice_2/stack_2ъ
<sequential_2/sequential_1/conv2d_transpose_1/strided_slice_2StridedSlice;sequential_2/sequential_1/conv2d_transpose_1/Shape:output:0Ksequential_2/sequential_1/conv2d_transpose_1/strided_slice_2/stack:output:0Msequential_2/sequential_1/conv2d_transpose_1/strided_slice_2/stack_1:output:0Msequential_2/sequential_1/conv2d_transpose_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_2/sequential_1/conv2d_transpose_1/strided_slice_2™
2sequential_2/sequential_1/conv2d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_2/sequential_1/conv2d_transpose_1/mul/yР
0sequential_2/sequential_1/conv2d_transpose_1/mulMulEsequential_2/sequential_1/conv2d_transpose_1/strided_slice_1:output:0;sequential_2/sequential_1/conv2d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 22
0sequential_2/sequential_1/conv2d_transpose_1/mul™
2sequential_2/sequential_1/conv2d_transpose_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 24
2sequential_2/sequential_1/conv2d_transpose_1/add/yБ
0sequential_2/sequential_1/conv2d_transpose_1/addAddV24sequential_2/sequential_1/conv2d_transpose_1/mul:z:0;sequential_2/sequential_1/conv2d_transpose_1/add/y:output:0*
T0*
_output_shapes
: 22
0sequential_2/sequential_1/conv2d_transpose_1/addЃ
4sequential_2/sequential_1/conv2d_transpose_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :26
4sequential_2/sequential_1/conv2d_transpose_1/mul_1/yЦ
2sequential_2/sequential_1/conv2d_transpose_1/mul_1MulEsequential_2/sequential_1/conv2d_transpose_1/strided_slice_2:output:0=sequential_2/sequential_1/conv2d_transpose_1/mul_1/y:output:0*
T0*
_output_shapes
: 24
2sequential_2/sequential_1/conv2d_transpose_1/mul_1Ѓ
4sequential_2/sequential_1/conv2d_transpose_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 26
4sequential_2/sequential_1/conv2d_transpose_1/add_1/yЙ
2sequential_2/sequential_1/conv2d_transpose_1/add_1AddV26sequential_2/sequential_1/conv2d_transpose_1/mul_1:z:0=sequential_2/sequential_1/conv2d_transpose_1/add_1/y:output:0*
T0*
_output_shapes
: 24
2sequential_2/sequential_1/conv2d_transpose_1/add_1Ѓ
4sequential_2/sequential_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@26
4sequential_2/sequential_1/conv2d_transpose_1/stack/3Р
2sequential_2/sequential_1/conv2d_transpose_1/stackPackCsequential_2/sequential_1/conv2d_transpose_1/strided_slice:output:04sequential_2/sequential_1/conv2d_transpose_1/add:z:06sequential_2/sequential_1/conv2d_transpose_1/add_1:z:0=sequential_2/sequential_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:24
2sequential_2/sequential_1/conv2d_transpose_1/stack“
Bsequential_2/sequential_1/conv2d_transpose_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_2/sequential_1/conv2d_transpose_1/strided_slice_3/stack÷
Dsequential_2/sequential_1/conv2d_transpose_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_1/strided_slice_3/stack_1÷
Dsequential_2/sequential_1/conv2d_transpose_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_1/strided_slice_3/stack_2ъ
<sequential_2/sequential_1/conv2d_transpose_1/strided_slice_3StridedSlice;sequential_2/sequential_1/conv2d_transpose_1/stack:output:0Ksequential_2/sequential_1/conv2d_transpose_1/strided_slice_3/stack:output:0Msequential_2/sequential_1/conv2d_transpose_1/strided_slice_3/stack_1:output:0Msequential_2/sequential_1/conv2d_transpose_1/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_2/sequential_1/conv2d_transpose_1/strided_slice_3Ї
Lsequential_2/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpUsequential_2_sequential_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02N
Lsequential_2/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp 
=sequential_2/sequential_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput;sequential_2/sequential_1/conv2d_transpose_1/stack:output:0Tsequential_2/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0=sequential_2/sequential_1/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@*
paddingVALID*
strides
2?
=sequential_2/sequential_1/conv2d_transpose_1/conv2d_transposeУ
Csequential_2/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpLsequential_2_sequential_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02E
Csequential_2/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp∆
4sequential_2/sequential_1/conv2d_transpose_1/BiasAddBiasAddFsequential_2/sequential_1/conv2d_transpose_1/conv2d_transpose:output:0Ksequential_2/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@26
4sequential_2/sequential_1/conv2d_transpose_1/BiasAddз
1sequential_2/sequential_1/conv2d_transpose_1/ReluRelu=sequential_2/sequential_1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€@23
1sequential_2/sequential_1/conv2d_transpose_1/Relu„
2sequential_2/sequential_1/conv2d_transpose_2/ShapeShape?sequential_2/sequential_1/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:24
2sequential_2/sequential_1/conv2d_transpose_2/Shapeќ
@sequential_2/sequential_1/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_2/sequential_1/conv2d_transpose_2/strided_slice/stack“
Bsequential_2/sequential_1/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose_2/strided_slice/stack_1“
Bsequential_2/sequential_1/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose_2/strided_slice/stack_2р
:sequential_2/sequential_1/conv2d_transpose_2/strided_sliceStridedSlice;sequential_2/sequential_1/conv2d_transpose_2/Shape:output:0Isequential_2/sequential_1/conv2d_transpose_2/strided_slice/stack:output:0Ksequential_2/sequential_1/conv2d_transpose_2/strided_slice/stack_1:output:0Ksequential_2/sequential_1/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_2/sequential_1/conv2d_transpose_2/strided_slice“
Bsequential_2/sequential_1/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose_2/strided_slice_1/stack÷
Dsequential_2/sequential_1/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_2/strided_slice_1/stack_1÷
Dsequential_2/sequential_1/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_2/strided_slice_1/stack_2ъ
<sequential_2/sequential_1/conv2d_transpose_2/strided_slice_1StridedSlice;sequential_2/sequential_1/conv2d_transpose_2/Shape:output:0Ksequential_2/sequential_1/conv2d_transpose_2/strided_slice_1/stack:output:0Msequential_2/sequential_1/conv2d_transpose_2/strided_slice_1/stack_1:output:0Msequential_2/sequential_1/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_2/sequential_1/conv2d_transpose_2/strided_slice_1“
Bsequential_2/sequential_1/conv2d_transpose_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose_2/strided_slice_2/stack÷
Dsequential_2/sequential_1/conv2d_transpose_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_2/strided_slice_2/stack_1÷
Dsequential_2/sequential_1/conv2d_transpose_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_2/strided_slice_2/stack_2ъ
<sequential_2/sequential_1/conv2d_transpose_2/strided_slice_2StridedSlice;sequential_2/sequential_1/conv2d_transpose_2/Shape:output:0Ksequential_2/sequential_1/conv2d_transpose_2/strided_slice_2/stack:output:0Msequential_2/sequential_1/conv2d_transpose_2/strided_slice_2/stack_1:output:0Msequential_2/sequential_1/conv2d_transpose_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_2/sequential_1/conv2d_transpose_2/strided_slice_2™
2sequential_2/sequential_1/conv2d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_2/sequential_1/conv2d_transpose_2/mul/yР
0sequential_2/sequential_1/conv2d_transpose_2/mulMulEsequential_2/sequential_1/conv2d_transpose_2/strided_slice_1:output:0;sequential_2/sequential_1/conv2d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 22
0sequential_2/sequential_1/conv2d_transpose_2/mul™
2sequential_2/sequential_1/conv2d_transpose_2/add/yConst*
_output_shapes
: *
dtype0*
value	B : 24
2sequential_2/sequential_1/conv2d_transpose_2/add/yБ
0sequential_2/sequential_1/conv2d_transpose_2/addAddV24sequential_2/sequential_1/conv2d_transpose_2/mul:z:0;sequential_2/sequential_1/conv2d_transpose_2/add/y:output:0*
T0*
_output_shapes
: 22
0sequential_2/sequential_1/conv2d_transpose_2/addЃ
4sequential_2/sequential_1/conv2d_transpose_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :26
4sequential_2/sequential_1/conv2d_transpose_2/mul_1/yЦ
2sequential_2/sequential_1/conv2d_transpose_2/mul_1MulEsequential_2/sequential_1/conv2d_transpose_2/strided_slice_2:output:0=sequential_2/sequential_1/conv2d_transpose_2/mul_1/y:output:0*
T0*
_output_shapes
: 24
2sequential_2/sequential_1/conv2d_transpose_2/mul_1Ѓ
4sequential_2/sequential_1/conv2d_transpose_2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 26
4sequential_2/sequential_1/conv2d_transpose_2/add_1/yЙ
2sequential_2/sequential_1/conv2d_transpose_2/add_1AddV26sequential_2/sequential_1/conv2d_transpose_2/mul_1:z:0=sequential_2/sequential_1/conv2d_transpose_2/add_1/y:output:0*
T0*
_output_shapes
: 24
2sequential_2/sequential_1/conv2d_transpose_2/add_1Ѓ
4sequential_2/sequential_1/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@26
4sequential_2/sequential_1/conv2d_transpose_2/stack/3Р
2sequential_2/sequential_1/conv2d_transpose_2/stackPackCsequential_2/sequential_1/conv2d_transpose_2/strided_slice:output:04sequential_2/sequential_1/conv2d_transpose_2/add:z:06sequential_2/sequential_1/conv2d_transpose_2/add_1:z:0=sequential_2/sequential_1/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:24
2sequential_2/sequential_1/conv2d_transpose_2/stack“
Bsequential_2/sequential_1/conv2d_transpose_2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_2/sequential_1/conv2d_transpose_2/strided_slice_3/stack÷
Dsequential_2/sequential_1/conv2d_transpose_2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_2/strided_slice_3/stack_1÷
Dsequential_2/sequential_1/conv2d_transpose_2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_2/strided_slice_3/stack_2ъ
<sequential_2/sequential_1/conv2d_transpose_2/strided_slice_3StridedSlice;sequential_2/sequential_1/conv2d_transpose_2/stack:output:0Ksequential_2/sequential_1/conv2d_transpose_2/strided_slice_3/stack:output:0Msequential_2/sequential_1/conv2d_transpose_2/strided_slice_3/stack_1:output:0Msequential_2/sequential_1/conv2d_transpose_2/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_2/sequential_1/conv2d_transpose_2/strided_slice_3Ї
Lsequential_2/sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpUsequential_2_sequential_1_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@@*
dtype02N
Lsequential_2/sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpћ
=sequential_2/sequential_1/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput;sequential_2/sequential_1/conv2d_transpose_2/stack:output:0Tsequential_2/sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0?sequential_2/sequential_1/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€0@*
paddingVALID*
strides
2?
=sequential_2/sequential_1/conv2d_transpose_2/conv2d_transposeУ
Csequential_2/sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpLsequential_2_sequential_1_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02E
Csequential_2/sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp∆
4sequential_2/sequential_1/conv2d_transpose_2/BiasAddBiasAddFsequential_2/sequential_1/conv2d_transpose_2/conv2d_transpose:output:0Ksequential_2/sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€0@26
4sequential_2/sequential_1/conv2d_transpose_2/BiasAddз
1sequential_2/sequential_1/conv2d_transpose_2/ReluRelu=sequential_2/sequential_1/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€0@23
1sequential_2/sequential_1/conv2d_transpose_2/Relu„
2sequential_2/sequential_1/conv2d_transpose_3/ShapeShape?sequential_2/sequential_1/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:24
2sequential_2/sequential_1/conv2d_transpose_3/Shapeќ
@sequential_2/sequential_1/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_2/sequential_1/conv2d_transpose_3/strided_slice/stack“
Bsequential_2/sequential_1/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose_3/strided_slice/stack_1“
Bsequential_2/sequential_1/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose_3/strided_slice/stack_2р
:sequential_2/sequential_1/conv2d_transpose_3/strided_sliceStridedSlice;sequential_2/sequential_1/conv2d_transpose_3/Shape:output:0Isequential_2/sequential_1/conv2d_transpose_3/strided_slice/stack:output:0Ksequential_2/sequential_1/conv2d_transpose_3/strided_slice/stack_1:output:0Ksequential_2/sequential_1/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_2/sequential_1/conv2d_transpose_3/strided_slice“
Bsequential_2/sequential_1/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose_3/strided_slice_1/stack÷
Dsequential_2/sequential_1/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_3/strided_slice_1/stack_1÷
Dsequential_2/sequential_1/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_3/strided_slice_1/stack_2ъ
<sequential_2/sequential_1/conv2d_transpose_3/strided_slice_1StridedSlice;sequential_2/sequential_1/conv2d_transpose_3/Shape:output:0Ksequential_2/sequential_1/conv2d_transpose_3/strided_slice_1/stack:output:0Msequential_2/sequential_1/conv2d_transpose_3/strided_slice_1/stack_1:output:0Msequential_2/sequential_1/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_2/sequential_1/conv2d_transpose_3/strided_slice_1“
Bsequential_2/sequential_1/conv2d_transpose_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_2/sequential_1/conv2d_transpose_3/strided_slice_2/stack÷
Dsequential_2/sequential_1/conv2d_transpose_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_3/strided_slice_2/stack_1÷
Dsequential_2/sequential_1/conv2d_transpose_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_3/strided_slice_2/stack_2ъ
<sequential_2/sequential_1/conv2d_transpose_3/strided_slice_2StridedSlice;sequential_2/sequential_1/conv2d_transpose_3/Shape:output:0Ksequential_2/sequential_1/conv2d_transpose_3/strided_slice_2/stack:output:0Msequential_2/sequential_1/conv2d_transpose_3/strided_slice_2/stack_1:output:0Msequential_2/sequential_1/conv2d_transpose_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_2/sequential_1/conv2d_transpose_3/strided_slice_2™
2sequential_2/sequential_1/conv2d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_2/sequential_1/conv2d_transpose_3/mul/yР
0sequential_2/sequential_1/conv2d_transpose_3/mulMulEsequential_2/sequential_1/conv2d_transpose_3/strided_slice_1:output:0;sequential_2/sequential_1/conv2d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 22
0sequential_2/sequential_1/conv2d_transpose_3/mul™
2sequential_2/sequential_1/conv2d_transpose_3/add/yConst*
_output_shapes
: *
dtype0*
value	B : 24
2sequential_2/sequential_1/conv2d_transpose_3/add/yБ
0sequential_2/sequential_1/conv2d_transpose_3/addAddV24sequential_2/sequential_1/conv2d_transpose_3/mul:z:0;sequential_2/sequential_1/conv2d_transpose_3/add/y:output:0*
T0*
_output_shapes
: 22
0sequential_2/sequential_1/conv2d_transpose_3/addЃ
4sequential_2/sequential_1/conv2d_transpose_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :26
4sequential_2/sequential_1/conv2d_transpose_3/mul_1/yЦ
2sequential_2/sequential_1/conv2d_transpose_3/mul_1MulEsequential_2/sequential_1/conv2d_transpose_3/strided_slice_2:output:0=sequential_2/sequential_1/conv2d_transpose_3/mul_1/y:output:0*
T0*
_output_shapes
: 24
2sequential_2/sequential_1/conv2d_transpose_3/mul_1Ѓ
4sequential_2/sequential_1/conv2d_transpose_3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 26
4sequential_2/sequential_1/conv2d_transpose_3/add_1/yЙ
2sequential_2/sequential_1/conv2d_transpose_3/add_1AddV26sequential_2/sequential_1/conv2d_transpose_3/mul_1:z:0=sequential_2/sequential_1/conv2d_transpose_3/add_1/y:output:0*
T0*
_output_shapes
: 24
2sequential_2/sequential_1/conv2d_transpose_3/add_1Ѓ
4sequential_2/sequential_1/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :26
4sequential_2/sequential_1/conv2d_transpose_3/stack/3Р
2sequential_2/sequential_1/conv2d_transpose_3/stackPackCsequential_2/sequential_1/conv2d_transpose_3/strided_slice:output:04sequential_2/sequential_1/conv2d_transpose_3/add:z:06sequential_2/sequential_1/conv2d_transpose_3/add_1:z:0=sequential_2/sequential_1/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:24
2sequential_2/sequential_1/conv2d_transpose_3/stack“
Bsequential_2/sequential_1/conv2d_transpose_3/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_2/sequential_1/conv2d_transpose_3/strided_slice_3/stack÷
Dsequential_2/sequential_1/conv2d_transpose_3/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_3/strided_slice_3/stack_1÷
Dsequential_2/sequential_1/conv2d_transpose_3/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_2/sequential_1/conv2d_transpose_3/strided_slice_3/stack_2ъ
<sequential_2/sequential_1/conv2d_transpose_3/strided_slice_3StridedSlice;sequential_2/sequential_1/conv2d_transpose_3/stack:output:0Ksequential_2/sequential_1/conv2d_transpose_3/strided_slice_3/stack:output:0Msequential_2/sequential_1/conv2d_transpose_3/strided_slice_3/stack_1:output:0Msequential_2/sequential_1/conv2d_transpose_3/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_2/sequential_1/conv2d_transpose_3/strided_slice_3Ї
Lsequential_2/sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpUsequential_2_sequential_1_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@*
dtype02N
Lsequential_2/sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpЌ
=sequential_2/sequential_1/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput;sequential_2/sequential_1/conv2d_transpose_3/stack:output:0Tsequential_2/sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0?sequential_2/sequential_1/conv2d_transpose_2/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€`ј*
paddingVALID*
strides
2?
=sequential_2/sequential_1/conv2d_transpose_3/conv2d_transposeУ
Csequential_2/sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpLsequential_2_sequential_1_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02E
Csequential_2/sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOp«
4sequential_2/sequential_1/conv2d_transpose_3/BiasAddBiasAddFsequential_2/sequential_1/conv2d_transpose_3/conv2d_transpose:output:0Ksequential_2/sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€`ј26
4sequential_2/sequential_1/conv2d_transpose_3/BiasAddи
1sequential_2/sequential_1/conv2d_transpose_3/ReluRelu=sequential_2/sequential_1/conv2d_transpose_3/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€`ј23
1sequential_2/sequential_1/conv2d_transpose_3/ReluЬ
IdentityIdentity?sequential_2/sequential_1/conv2d_transpose_3/Relu:activations:06^sequential_2/sequential/conv2d/BiasAdd/ReadVariableOp5^sequential_2/sequential/conv2d/Conv2D/ReadVariableOp8^sequential_2/sequential/conv2d_1/BiasAdd/ReadVariableOp7^sequential_2/sequential/conv2d_1/Conv2D/ReadVariableOp8^sequential_2/sequential/conv2d_2/BiasAdd/ReadVariableOp7^sequential_2/sequential/conv2d_2/Conv2D/ReadVariableOp8^sequential_2/sequential/conv2d_3/BiasAdd/ReadVariableOp7^sequential_2/sequential/conv2d_3/Conv2D/ReadVariableOp5^sequential_2/sequential/dense/BiasAdd/ReadVariableOp7^sequential_2/sequential/dense/Tensordot/ReadVariableOpB^sequential_2/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpK^sequential_2/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpD^sequential_2/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpM^sequential_2/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpD^sequential_2/sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpM^sequential_2/sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpD^sequential_2/sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOpM^sequential_2/sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp9^sequential_2/sequential_1/dense_1/BiasAdd/ReadVariableOp;^sequential_2/sequential_1/dense_1/Tensordot/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€`ј2

Identity"
identityIdentity:output:0*
_input_shapesn
l:€€€€€€€€€`ј::::::::::::::::::::2n
5sequential_2/sequential/conv2d/BiasAdd/ReadVariableOp5sequential_2/sequential/conv2d/BiasAdd/ReadVariableOp2l
4sequential_2/sequential/conv2d/Conv2D/ReadVariableOp4sequential_2/sequential/conv2d/Conv2D/ReadVariableOp2r
7sequential_2/sequential/conv2d_1/BiasAdd/ReadVariableOp7sequential_2/sequential/conv2d_1/BiasAdd/ReadVariableOp2p
6sequential_2/sequential/conv2d_1/Conv2D/ReadVariableOp6sequential_2/sequential/conv2d_1/Conv2D/ReadVariableOp2r
7sequential_2/sequential/conv2d_2/BiasAdd/ReadVariableOp7sequential_2/sequential/conv2d_2/BiasAdd/ReadVariableOp2p
6sequential_2/sequential/conv2d_2/Conv2D/ReadVariableOp6sequential_2/sequential/conv2d_2/Conv2D/ReadVariableOp2r
7sequential_2/sequential/conv2d_3/BiasAdd/ReadVariableOp7sequential_2/sequential/conv2d_3/BiasAdd/ReadVariableOp2p
6sequential_2/sequential/conv2d_3/Conv2D/ReadVariableOp6sequential_2/sequential/conv2d_3/Conv2D/ReadVariableOp2l
4sequential_2/sequential/dense/BiasAdd/ReadVariableOp4sequential_2/sequential/dense/BiasAdd/ReadVariableOp2p
6sequential_2/sequential/dense/Tensordot/ReadVariableOp6sequential_2/sequential/dense/Tensordot/ReadVariableOp2Ж
Asequential_2/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOpAsequential_2/sequential_1/conv2d_transpose/BiasAdd/ReadVariableOp2Ш
Jsequential_2/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOpJsequential_2/sequential_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2К
Csequential_2/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOpCsequential_2/sequential_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2Ь
Lsequential_2/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpLsequential_2/sequential_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2К
Csequential_2/sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOpCsequential_2/sequential_1/conv2d_transpose_2/BiasAdd/ReadVariableOp2Ь
Lsequential_2/sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOpLsequential_2/sequential_1/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2К
Csequential_2/sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOpCsequential_2/sequential_1/conv2d_transpose_3/BiasAdd/ReadVariableOp2Ь
Lsequential_2/sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOpLsequential_2/sequential_1/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2t
8sequential_2/sequential_1/dense_1/BiasAdd/ReadVariableOp8sequential_2/sequential_1/dense_1/BiasAdd/ReadVariableOp2x
:sequential_2/sequential_1/dense_1/Tensordot/ReadVariableOp:sequential_2/sequential_1/dense_1/Tensordot/ReadVariableOp:' #
!
_user_specified_name	input_1"ѓL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*љ
serving_default©
D
input_19
serving_default_input_1:0€€€€€€€€€`јE
output_19
StatefulPartitionedCall:0€€€€€€€€€`јtensorflow/serving/predict:ќГ
Я`
layer-0
layer-1
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+ƒ&call_and_return_all_conditional_losses
≈__call__
∆_default_save_signature"Ј^
_tf_keras_sequentialШ^{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_2", "layers": [{"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 4], "strides": [3, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [2, 3], "strides": [2, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [2, 3], "strides": [2, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 4], "strides": [3, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}}], "build_input_shape": [null, 96, 192, 1]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 4], "strides": [3, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [2, 3], "strides": [2, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [2, 3], "strides": [2, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 4], "strides": [3, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}}], "build_input_shape": [null, 96, 192, 1]}}, "training_config": {"loss": "mse", "metrics": ["mse"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
џ.
	layer-0

layer-1
layer-2
layer-3
layer-4
regularization_losses
trainable_variables
	variables
	keras_api
+«&call_and_return_all_conditional_losses
»__call__"Й-
_tf_keras_sequentialк,{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 4], "strides": [3, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [2, 3], "strides": [2, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 4], "strides": [3, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [2, 3], "strides": [2, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
Љ1
layer-0
layer-1
layer-2
layer-3
layer-4
regularization_losses
trainable_variables
	variables
	keras_api
+…&call_and_return_all_conditional_losses
 __call__"к/
_tf_keras_sequentialЋ/{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [2, 3], "strides": [2, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 4], "strides": [3, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [2, 3], "strides": [2, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 4], "strides": [3, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}}]}}}
г
iter

beta_1

beta_2
	decay
learning_rate mЬ!mЭ"mЮ#mЯ$m†%m°&mҐ'm£(m§)m•*m¶+mІ,m®-m©.m™/mЂ0mђ1m≠2mЃ3mѓ v∞!v±"v≤#v≥$vі%vµ&vґ'vЈ(vЄ)vє*vЇ+vї,vЉ-vљ.vЊ/vњ0vј1vЅ2v¬3v√"
	optimizer
 "
trackable_list_wrapper
ґ
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14
/15
016
117
218
319"
trackable_list_wrapper
ґ
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
,12
-13
.14
/15
016
117
218
319"
trackable_list_wrapper
ї
4metrics

5layers
regularization_losses
trainable_variables
6non_trainable_variables
7layer_regularization_losses
	variables
≈__call__
∆_default_save_signature
+ƒ&call_and_return_all_conditional_losses
'ƒ"call_and_return_conditional_losses"
_generic_user_object
-
Ћserving_default"
signature_map
к

 kernel
!bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
+ћ&call_and_return_all_conditional_losses
Ќ__call__"√
_tf_keras_layer©{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
п

"kernel
#bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
+ќ&call_and_return_all_conditional_losses
ѕ__call__"»
_tf_keras_layerЃ{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
п

$kernel
%bias
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
+–&call_and_return_all_conditional_losses
—__call__"»
_tf_keras_layerЃ{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 4], "strides": [3, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
р

&kernel
'bias
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
+“&call_and_return_all_conditional_losses
”__call__"…
_tf_keras_layerѓ{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [2, 3], "strides": [2, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
т

(kernel
)bias
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
+‘&call_and_return_all_conditional_losses
’__call__"Ћ
_tf_keras_layer±{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
 "
trackable_list_wrapper
f
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9"
trackable_list_wrapper
f
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9"
trackable_list_wrapper
Э
Lmetrics

Mlayers
regularization_losses
trainable_variables
Nnon_trainable_variables
Olayer_regularization_losses
	variables
»__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
ц

*kernel
+bias
Pregularization_losses
Qtrainable_variables
R	variables
S	keras_api
+÷&call_and_return_all_conditional_losses
„__call__"ѕ
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}}
°

,kernel
-bias
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
+Ў&call_and_return_all_conditional_losses
ў__call__"ъ
_tf_keras_layerа{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [2, 3], "strides": [2, 3], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
§

.kernel
/bias
Xregularization_losses
Ytrainable_variables
Z	variables
[	keras_api
+Џ&call_and_return_all_conditional_losses
џ__call__"э
_tf_keras_layerг{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 4], "strides": [3, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
§

0kernel
1bias
\regularization_losses
]trainable_variables
^	variables
_	keras_api
+№&call_and_return_all_conditional_losses
Ё__call__"э
_tf_keras_layerг{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
£

2kernel
3bias
`regularization_losses
atrainable_variables
b	variables
c	keras_api
+ё&call_and_return_all_conditional_losses
я__call__"ь
_tf_keras_layerв{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": [4, 4], "strides": [4, 4], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
 "
trackable_list_wrapper
f
*0
+1
,2
-3
.4
/5
06
17
28
39"
trackable_list_wrapper
f
*0
+1
,2
-3
.4
/5
06
17
28
39"
trackable_list_wrapper
Э
dmetrics

elayers
regularization_losses
trainable_variables
fnon_trainable_variables
glayer_regularization_losses
	variables
 __call__
+…&call_and_return_all_conditional_losses
'…"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?:=@2%sequential_2/sequential/conv2d/kernel
1:/@2#sequential_2/sequential/conv2d/bias
A:?@@2'sequential_2/sequential/conv2d_1/kernel
3:1@2%sequential_2/sequential/conv2d_1/bias
A:?@@2'sequential_2/sequential/conv2d_2/kernel
3:1@2%sequential_2/sequential/conv2d_2/bias
B:@@А2'sequential_2/sequential/conv2d_3/kernel
4:2А2%sequential_2/sequential/conv2d_3/bias
7:5	А2$sequential_2/sequential/dense/kernel
0:.2"sequential_2/sequential/dense/bias
;:9	А2(sequential_2/sequential_1/dense_1/kernel
5:3А2&sequential_2/sequential_1/dense_1/bias
L:J@А21sequential_2/sequential_1/conv2d_transpose/kernel
=:;@2/sequential_2/sequential_1/conv2d_transpose/bias
M:K@@23sequential_2/sequential_1/conv2d_transpose_1/kernel
?:=@21sequential_2/sequential_1/conv2d_transpose_1/bias
M:K@@23sequential_2/sequential_1/conv2d_transpose_2/kernel
?:=@21sequential_2/sequential_1/conv2d_transpose_2/bias
M:K@23sequential_2/sequential_1/conv2d_transpose_3/kernel
?:=21sequential_2/sequential_1/conv2d_transpose_3/bias
'
h0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
Э
imetrics

jlayers
8regularization_losses
9trainable_variables
knon_trainable_variables
llayer_regularization_losses
:	variables
Ќ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
Э
mmetrics

nlayers
<regularization_losses
=trainable_variables
onon_trainable_variables
player_regularization_losses
>	variables
ѕ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
Э
qmetrics

rlayers
@regularization_losses
Atrainable_variables
snon_trainable_variables
tlayer_regularization_losses
B	variables
—__call__
+–&call_and_return_all_conditional_losses
'–"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
Э
umetrics

vlayers
Dregularization_losses
Etrainable_variables
wnon_trainable_variables
xlayer_regularization_losses
F	variables
”__call__
+“&call_and_return_all_conditional_losses
'“"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
Э
ymetrics

zlayers
Hregularization_losses
Itrainable_variables
{non_trainable_variables
|layer_regularization_losses
J	variables
’__call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
C
	0

1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
Ю
}metrics

~layers
Pregularization_losses
Qtrainable_variables
non_trainable_variables
 Аlayer_regularization_losses
R	variables
„__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
°
Бmetrics
Вlayers
Tregularization_losses
Utrainable_variables
Гnon_trainable_variables
 Дlayer_regularization_losses
V	variables
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
°
Еmetrics
Жlayers
Xregularization_losses
Ytrainable_variables
Зnon_trainable_variables
 Иlayer_regularization_losses
Z	variables
џ__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
°
Йmetrics
Кlayers
\regularization_losses
]trainable_variables
Лnon_trainable_variables
 Мlayer_regularization_losses
^	variables
Ё__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
°
Нmetrics
Оlayers
`regularization_losses
atrainable_variables
Пnon_trainable_variables
 Рlayer_regularization_losses
b	variables
я__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Щ

Сtotal

Тcount
У
_fn_kwargs
Фregularization_losses
Хtrainable_variables
Ц	variables
Ч	keras_api
+а&call_and_return_all_conditional_losses
б__call__"џ
_tf_keras_layerЅ{"class_name": "MeanMetricWrapper", "name": "mse", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mse", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
С0
Т1"
trackable_list_wrapper
§
Шmetrics
Щlayers
Фregularization_losses
Хtrainable_variables
Ъnon_trainable_variables
 Ыlayer_regularization_losses
Ц	variables
б__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
С0
Т1"
trackable_list_wrapper
 "
trackable_list_wrapper
D:B@2,Adam/sequential_2/sequential/conv2d/kernel/m
6:4@2*Adam/sequential_2/sequential/conv2d/bias/m
F:D@@2.Adam/sequential_2/sequential/conv2d_1/kernel/m
8:6@2,Adam/sequential_2/sequential/conv2d_1/bias/m
F:D@@2.Adam/sequential_2/sequential/conv2d_2/kernel/m
8:6@2,Adam/sequential_2/sequential/conv2d_2/bias/m
G:E@А2.Adam/sequential_2/sequential/conv2d_3/kernel/m
9:7А2,Adam/sequential_2/sequential/conv2d_3/bias/m
<::	А2+Adam/sequential_2/sequential/dense/kernel/m
5:32)Adam/sequential_2/sequential/dense/bias/m
@:>	А2/Adam/sequential_2/sequential_1/dense_1/kernel/m
::8А2-Adam/sequential_2/sequential_1/dense_1/bias/m
Q:O@А28Adam/sequential_2/sequential_1/conv2d_transpose/kernel/m
B:@@26Adam/sequential_2/sequential_1/conv2d_transpose/bias/m
R:P@@2:Adam/sequential_2/sequential_1/conv2d_transpose_1/kernel/m
D:B@28Adam/sequential_2/sequential_1/conv2d_transpose_1/bias/m
R:P@@2:Adam/sequential_2/sequential_1/conv2d_transpose_2/kernel/m
D:B@28Adam/sequential_2/sequential_1/conv2d_transpose_2/bias/m
R:P@2:Adam/sequential_2/sequential_1/conv2d_transpose_3/kernel/m
D:B28Adam/sequential_2/sequential_1/conv2d_transpose_3/bias/m
D:B@2,Adam/sequential_2/sequential/conv2d/kernel/v
6:4@2*Adam/sequential_2/sequential/conv2d/bias/v
F:D@@2.Adam/sequential_2/sequential/conv2d_1/kernel/v
8:6@2,Adam/sequential_2/sequential/conv2d_1/bias/v
F:D@@2.Adam/sequential_2/sequential/conv2d_2/kernel/v
8:6@2,Adam/sequential_2/sequential/conv2d_2/bias/v
G:E@А2.Adam/sequential_2/sequential/conv2d_3/kernel/v
9:7А2,Adam/sequential_2/sequential/conv2d_3/bias/v
<::	А2+Adam/sequential_2/sequential/dense/kernel/v
5:32)Adam/sequential_2/sequential/dense/bias/v
@:>	А2/Adam/sequential_2/sequential_1/dense_1/kernel/v
::8А2-Adam/sequential_2/sequential_1/dense_1/bias/v
Q:O@А28Adam/sequential_2/sequential_1/conv2d_transpose/kernel/v
B:@@26Adam/sequential_2/sequential_1/conv2d_transpose/bias/v
R:P@@2:Adam/sequential_2/sequential_1/conv2d_transpose_1/kernel/v
D:B@28Adam/sequential_2/sequential_1/conv2d_transpose_1/bias/v
R:P@@2:Adam/sequential_2/sequential_1/conv2d_transpose_2/kernel/v
D:B@28Adam/sequential_2/sequential_1/conv2d_transpose_2/bias/v
R:P@2:Adam/sequential_2/sequential_1/conv2d_transpose_3/kernel/v
D:B28Adam/sequential_2/sequential_1/conv2d_transpose_3/bias/v
к2з
G__inference_sequential_2_layer_call_and_return_conditional_losses_37445
G__inference_sequential_2_layer_call_and_return_conditional_losses_37217
G__inference_sequential_2_layer_call_and_return_conditional_losses_36852
G__inference_sequential_2_layer_call_and_return_conditional_losses_36826ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
,__inference_sequential_2_layer_call_fn_37470
,__inference_sequential_2_layer_call_fn_36955
,__inference_sequential_2_layer_call_fn_36904
,__inference_sequential_2_layer_call_fn_37495ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
з2д
 __inference__wrapped_model_36165њ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ */Ґ,
*К'
input_1€€€€€€€€€`ј
в2я
E__inference_sequential_layer_call_and_return_conditional_losses_36312
E__inference_sequential_layer_call_and_return_conditional_losses_37619
E__inference_sequential_layer_call_and_return_conditional_losses_36331
E__inference_sequential_layer_call_and_return_conditional_losses_37557ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ц2у
*__inference_sequential_layer_call_fn_36366
*__inference_sequential_layer_call_fn_36400
*__inference_sequential_layer_call_fn_37634
*__inference_sequential_layer_call_fn_37649ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
к2з
G__inference_sequential_1_layer_call_and_return_conditional_losses_37989
G__inference_sequential_1_layer_call_and_return_conditional_losses_37819
G__inference_sequential_1_layer_call_and_return_conditional_losses_36670
G__inference_sequential_1_layer_call_and_return_conditional_losses_36651ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
,__inference_sequential_1_layer_call_fn_38019
,__inference_sequential_1_layer_call_fn_36739
,__inference_sequential_1_layer_call_fn_38004
,__inference_sequential_1_layer_call_fn_36705ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
2B0
#__inference_signature_wrapper_36989input_1
†2Э
A__inference_conv2d_layer_call_and_return_conditional_losses_36178„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Е2В
&__inference_conv2d_layer_call_fn_36186„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ґ2Я
C__inference_conv2d_1_layer_call_and_return_conditional_losses_36199„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
З2Д
(__inference_conv2d_1_layer_call_fn_36207„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ґ2Я
C__inference_conv2d_2_layer_call_and_return_conditional_losses_36220„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
З2Д
(__inference_conv2d_2_layer_call_fn_36228„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ґ2Я
C__inference_conv2d_3_layer_call_and_return_conditional_losses_36241„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
З2Д
(__inference_conv2d_3_layer_call_fn_36249„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
к2з
@__inference_dense_layer_call_and_return_conditional_losses_38053Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
%__inference_dense_layer_call_fn_38060Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_1_layer_call_and_return_conditional_losses_38094Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_1_layer_call_fn_38101Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ђ2®
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_36439Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Р2Н
0__inference_conv2d_transpose_layer_call_fn_36447Ў
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *8Ґ5
3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
ђ2©
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_36486„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
С2О
2__inference_conv2d_transpose_1_layer_call_fn_36494„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
ђ2©
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_36533„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
С2О
2__inference_conv2d_transpose_2_layer_call_fn_36541„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
ђ2©
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_36580„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
С2О
2__inference_conv2d_transpose_3_layer_call_fn_36588„
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 
ћ2…∆
љ≤є
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaults™

trainingp 
annotations™ *
 і
 __inference__wrapped_model_36165П !"#$%&'()*+,-./01239Ґ6
/Ґ,
*К'
input_1€€€€€€€€€`ј
™ "<™9
7
output_1+К(
output_1€€€€€€€€€`јЎ
C__inference_conv2d_1_layer_call_and_return_conditional_losses_36199Р"#IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ∞
(__inference_conv2d_1_layer_call_fn_36207Г"#IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Ў
C__inference_conv2d_2_layer_call_and_return_conditional_losses_36220Р$%IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ∞
(__inference_conv2d_2_layer_call_fn_36228Г$%IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@ў
C__inference_conv2d_3_layer_call_and_return_conditional_losses_36241С&'IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ ±
(__inference_conv2d_3_layer_call_fn_36249Д&'IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А÷
A__inference_conv2d_layer_call_and_return_conditional_losses_36178Р !IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ Ѓ
&__inference_conv2d_layer_call_fn_36186Г !IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@в
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_36486Р./IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ Ї
2__inference_conv2d_transpose_1_layer_call_fn_36494Г./IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@в
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_36533Р01IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ Ї
2__inference_conv2d_transpose_2_layer_call_fn_36541Г01IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@в
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_36580Р23IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ї
2__inference_conv2d_transpose_3_layer_call_fn_36588Г23IҐF
?Ґ<
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€б
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_36439С,-JҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ є
0__inference_conv2d_transpose_layer_call_fn_36447Д,-JҐG
@Ґ=
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@≥
B__inference_dense_1_layer_call_and_return_conditional_losses_38094m*+7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Л
'__inference_dense_1_layer_call_fn_38101`*+7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "!К€€€€€€€€€А±
@__inference_dense_layer_call_and_return_conditional_losses_38053m()8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Й
%__inference_dense_layer_call_fn_38060`()8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ " К€€€€€€€€€џ
G__inference_sequential_1_layer_call_and_return_conditional_losses_36651П
*+,-./0123@Ґ=
6Ґ3
)К&
input_1€€€€€€€€€
p

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ џ
G__inference_sequential_1_layer_call_and_return_conditional_losses_36670П
*+,-./0123@Ґ=
6Ґ3
)К&
input_1€€€€€€€€€
p 

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ »
G__inference_sequential_1_layer_call_and_return_conditional_losses_37819}
*+,-./0123?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ ".Ґ+
$К!
0€€€€€€€€€`ј
Ъ »
G__inference_sequential_1_layer_call_and_return_conditional_losses_37989}
*+,-./0123?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ ".Ґ+
$К!
0€€€€€€€€€`ј
Ъ ≥
,__inference_sequential_1_layer_call_fn_36705В
*+,-./0123@Ґ=
6Ґ3
)К&
input_1€€€€€€€€€
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€≥
,__inference_sequential_1_layer_call_fn_36739В
*+,-./0123@Ґ=
6Ґ3
)К&
input_1€€€€€€€€€
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€≤
,__inference_sequential_1_layer_call_fn_38004Б
*+,-./0123?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€≤
,__inference_sequential_1_layer_call_fn_38019Б
*+,-./0123?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ж
G__inference_sequential_2_layer_call_and_return_conditional_losses_36826Ъ !"#$%&'()*+,-./0123AҐ>
7Ґ4
*К'
input_1€€€€€€€€€`ј
p

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ж
G__inference_sequential_2_layer_call_and_return_conditional_losses_36852Ъ !"#$%&'()*+,-./0123AҐ>
7Ґ4
*К'
input_1€€€€€€€€€`ј
p 

 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ‘
G__inference_sequential_2_layer_call_and_return_conditional_losses_37217И !"#$%&'()*+,-./0123@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€`ј
p

 
™ ".Ґ+
$К!
0€€€€€€€€€`ј
Ъ ‘
G__inference_sequential_2_layer_call_and_return_conditional_losses_37445И !"#$%&'()*+,-./0123@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€`ј
p 

 
™ ".Ґ+
$К!
0€€€€€€€€€`ј
Ъ Њ
,__inference_sequential_2_layer_call_fn_36904Н !"#$%&'()*+,-./0123AҐ>
7Ґ4
*К'
input_1€€€€€€€€€`ј
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€Њ
,__inference_sequential_2_layer_call_fn_36955Н !"#$%&'()*+,-./0123AҐ>
7Ґ4
*К'
input_1€€€€€€€€€`ј
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€љ
,__inference_sequential_2_layer_call_fn_37470М !"#$%&'()*+,-./0123@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€`ј
p

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€љ
,__inference_sequential_2_layer_call_fn_37495М !"#$%&'()*+,-./0123@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€`ј
p 

 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€«
E__inference_sequential_layer_call_and_return_conditional_losses_36312~
 !"#$%&'()AҐ>
7Ґ4
*К'
input_1€€€€€€€€€`ј
p

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ «
E__inference_sequential_layer_call_and_return_conditional_losses_36331~
 !"#$%&'()AҐ>
7Ґ4
*К'
input_1€€€€€€€€€`ј
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ ∆
E__inference_sequential_layer_call_and_return_conditional_losses_37557}
 !"#$%&'()@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€`ј
p

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ ∆
E__inference_sequential_layer_call_and_return_conditional_losses_37619}
 !"#$%&'()@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€`ј
p 

 
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Я
*__inference_sequential_layer_call_fn_36366q
 !"#$%&'()AҐ>
7Ґ4
*К'
input_1€€€€€€€€€`ј
p

 
™ " К€€€€€€€€€Я
*__inference_sequential_layer_call_fn_36400q
 !"#$%&'()AҐ>
7Ґ4
*К'
input_1€€€€€€€€€`ј
p 

 
™ " К€€€€€€€€€Ю
*__inference_sequential_layer_call_fn_37634p
 !"#$%&'()@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€`ј
p

 
™ " К€€€€€€€€€Ю
*__inference_sequential_layer_call_fn_37649p
 !"#$%&'()@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€`ј
p 

 
™ " К€€€€€€€€€¬
#__inference_signature_wrapper_36989Ъ !"#$%&'()*+,-./0123DҐA
Ґ 
:™7
5
input_1*К'
input_1€€€€€€€€€`ј"<™9
7
output_1+К(
output_1€€€€€€€€€`ј