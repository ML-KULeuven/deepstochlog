dom_number(X) :- member(X, [0,1,2,3,4,5,6,7,8,9]).
nn(number, [X], Y, dom_number) :: is_number(Y) --> [X].

dom_operator(X) :- member(X, [plus, minus, times, div]).
nn(operator, [X], Y, dom_operator) :: operator(Y) --> [X].
factor(N) --> is_number(N).

0.34 :: term(N) --> factor(N).
0.33 :: term(N) --> term(N1), operator(times), factor(N2), {N is N1 * N2}.
0.33 :: term(N) --> term(N1), operator(div), factor(N2), {N2>0, N is N1 / N2}.

0.34 :: expression(N) --> term(N).
0.33 :: expression(N) --> expression(N1), operator(plus), term(N2), {N is N1 + N2}.
0.33 :: expression(N) --> expression(N1), operator(minus), term(N2), {N is N1 - N2}.