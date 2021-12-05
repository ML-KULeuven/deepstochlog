neural(X,Y,_) -->  [], {nn(classifier, [X], Y), domain(Y, [class0,class1,class2,class3,class4,class5])}.
citep(X,Y,N) --> {N>0, N1 is N - 1, cite(X,X1)}, doc(X1,Y,N1).
doc(X,Y,N) --> {member(Z, [neural(X,Y,N),citep(X,Y,N)]), nn(rule_weight, [Y], Z)}, Z.
s(Y) --> [X], doc(X,Y,2).









