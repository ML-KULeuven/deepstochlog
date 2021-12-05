doc_neural(X,Y) -->  [], {nn(classifier, [X], Y), domain(Y, [class0,class1,class2,class3,class4,class5])}.
influence(X,Y) --> [], {cite(X,Y), findall(T, cite(X,T), L), nn(influence, [X,Y,L], 1)}.


doc(X,Y,N) --> doc_switch(X,Y,N,Z), {nn(rule_weight, [Y], Z), domain(Z, [neural,cite])}.
doc_switch(X,Y,_, neural) --> doc_neural(X,Y).
doc_switch(X,Y,N, cite) --> {N>0, N1 is N - 1}, influence(X, X1), doc(X1,Y,N1).

s(X) --> doc(X,Y,2), [Y].









