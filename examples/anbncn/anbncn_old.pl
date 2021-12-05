s --> rep(N,A), rep(N,B), rep(N,C),{A\=B, B\=C, C\=A}.
rep(0, _) --> [].
rep(s(N), C) --> [X], rep(N,C), {domain(C, [a,b,c]), nn(mnist, X, C)}.