dom_class(Y) :- member(Y, [class0,class1,class2,class3,class4,class5]).
neural_or_cite(Y) :- member(Y, [neural,cite]).
P::citep(X,Y) --> [], {cite(X,Y), findall(T, cite(X,T), L), length(L,M), P is 1 / M}.
nn(classifier, [X], Y, dom_class) :: doc_neural(X,Y) -->  [].
0.5::doc(X,Y,N) --> doc_neural(X,Y).
0.5::doc(X,Y,N) --> {N>0, N1 is N - 1}, citep(X,X1), doc(X1,Y,N1).
s(Y) --> {dom_class(Y)}, [X], doc(X,Y,2).










