OPENQASM 2.0;

OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];

h q[0];
t q[0];
h q[0];
h q[2];
s q[0];
cx q[2],q[1];
cx q[0],q[1];
h q[0];
