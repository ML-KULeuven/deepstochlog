letter(X) :- member(X, [a,b,c]).

0.5 :: s(0) --> akblcm(K,L,M),
                {K\=L; L\=M; M\=K},
                {K \= 0, L \= 0, M \= 0}.
0.5 :: s(1) --> akblcm(N,N,N).

akblcm(K,L,M) --> rep(K,A),
                  rep(L,B),
                  rep(M,C),
                  {A\=B, B\=C, C\=A}.

rep(0, _) --> [].
nn(mnist, [X], C, letter) :: rep(s(N), C) --> [X],
                                              rep(N,C).
