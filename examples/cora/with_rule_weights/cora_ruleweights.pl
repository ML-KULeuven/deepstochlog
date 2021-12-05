dom_class(X) :- member(X, [class0,class1,class2,class3,class4,class5,class6]).
nn(classifier, [X], Y, dom_class) :: doc_neural(X,Y) -->  [].

citep(X,Y) --> [], {cite(X,Y), findall(T, cite(X,T), L), length(L,M), P is 1 / M, p(P)}.

dom_rule_weight(X) :- member(X, [neural,cite]).
nn(rule_weight, [Y], Z, dom_rule_weight) :: doc(X,Y,N) --> doc_switch(X,Y,N,Z).
doc_switch(X,Y,_, neural) --> doc_neural(X,Y).
doc_switch(X,Y,N, cite) --> {N>0, N1 is N - 1}, citep(X, X1), doc(X1,Y,N1).
s(Y) --> [X], doc(X,Y,2).









