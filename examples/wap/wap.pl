dom_permute(X) :- member(X, [0,1,2,3,4,5]).
dom_op1(X) :- member(X, [plus,minus,times,div]).
dom_swap(X) :- member(X,[no_swap,swap]).
dom_op2(X) :- member(X, [plus,minus,times,div]).

nn(nn_permute, [Embed, Perm], Perm, dom_permute) :: nn_permute(Embed,Perm) --> [].
nn(nn_op1, [Embed], Op1, dom_op1) :: nn_op1(Embed, Op1) --> [].
nn(nn_swap, [Embed], Swap, dom_swap) :: nn_swap(Embed, Swap) --> [].
nn(nn_op2, [Embed], Op2, dom_op2) :: nn_op2(Embed, Op2) --> [].

0.1666666 :: permute(0,A,B,C,A,B,C) --> [].
0.1666666 :: permute(1,A,B,C,A,C,B) --> [].
0.1666666 :: permute(2,A,B,C,B,A,C) --> [].
0.1666666 :: permute(3,A,B,C,B,C,A) --> [].
0.1666666 :: permute(4,A,B,C,C,A,B) --> [].
0.1666666 :: permute(5,A,B,C,C,B,A) --> [].

0.5 :: swap(no_swap,X,Y,X,Y) --> [].
0.5 :: swap(swap,X,Y,Y,X) --> [].

0.25 :: operator(plus,X,Y,Z) --> [], {Z is X+Y}.
0.25 :: operator(minus,X,Y,Z) --> [], {Z is X-Y}.
0.25 :: operator(times,X,Y,Z) --> [], {Z is X*Y}.
0.25 :: operator(div,X,Y,Z) --> [], {Y > 0, 0 =:= X mod Y, Z is X//Y}.

s(Out,X1,X2,X3) --> [String],
                    nn_permute(String, Perm),
                    nn_op1(String, Op1),
                    nn_swap(String, Swap),
                    nn_op2(String, Op2),
                    permute(Perm,X1,X2,X3,N1,N2,N3),
                    operator(Op1,N1,N2,Res1),
                    swap(Swap,Res1,N3,X,Y),
                    operator(Op2,X,Y,Out).