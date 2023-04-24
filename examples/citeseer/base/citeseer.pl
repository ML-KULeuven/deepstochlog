doc_neural(X,Y) -->  [], {nn(classifier, [X], Y), domain(Y, [0,1,2,3,4,5])}.
citep(X,Y) --> [], {cite(X,Y), findall(T, cite(X,T), L), length(L,M), P is 1 / M, p(P)}.
doc(X,Y,_) --> doc_neural(X,Y), {p(0.5)}.
doc(X,Y,N) --> {N>0, N1 is N - 1, p(0.5)}, citep(X, X1), doc(X1,Y,N1).
s(X) --> doc(X,Y,1), [Y].




