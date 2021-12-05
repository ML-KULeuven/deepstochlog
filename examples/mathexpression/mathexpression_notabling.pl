is_number(Y) --> [X], {domain(Y,[0,1,2,3,4,5,6,7,8,9]), nn(number, Y)}.
operator(Y) --> [X], {domain(Y, [plus, minus, times, div]), nn(operator, Y)}.
term(N) --> is_number(N), {p(0.34)}.
term(N) --> is_number(N1), operator(times), term(N2), {N is N1 * N2, p(0.33)}.
term(N) --> is_number(N1), operator(div), term(N2), {N2>0, N is N1 / N2,p(0.33)}.
expression(N) --> term(N),{p(0.34)}.
expression(N) --> term(N1), operator(plus), expression(N2), {N is N1 + N2, p(0.33)}.
expression(N) --> term(N1), operator(minus), expression(N2), {N is N1 - N2, p(0.33)}.