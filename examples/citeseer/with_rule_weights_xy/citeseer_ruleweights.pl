doc_neural(X,Y) -->  [], {nn(classifier, X, Y), domain(Y, [class0,class1,class2,class3,class4,class5])}.
citep(X,Y) --> [], {cite(X,Y), findall(T, cite(X,T), L), length(L,M), P is 1 / M, p(P)}.


doc(X,Y,N) --> doc_switch(X,Y,N,Z), {nn(rule_weight, Y, Z), domain(Z, [neural,cite])}.

doc_switch(X,Y,_, neural) --> doc_neural(X,Y).
doc_switch(X,Y,N, cite) --> {N>0, N1 is N - 1, domain(Z, [class0,class1,class2,class3,class4,class5]), member(Y, [class0,class1,class2,class3,class4,class5]), nn(xy_switch, Y, Z)}, citep(X, X1), doc(X1,Z,N1).

s(X) --> doc(X,Y,1), [Y].









