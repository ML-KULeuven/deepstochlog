is_number(Y) --> [X], {domain(Y,[0,1,2,3,4,5,6,7,8,9]), nn(number,Y)}.
operator(Y) --> [X], {domain(Y, [plus, minus, times, div]), nn(operator, Y)}.
factor(N) --> is_number(N).
term(N) --> term_switch(N,Y), {nn(term, Y), domain(Y, [0,1,2])}.
term_switch(N, 0) --> factor(N).
term_switch(N, 1) --> term(N1), operator(times), factor(N2), {N is N1 * N2}.
term_switch(N, 2) --> term(N1), operator(div), factor(N2), {N2>0, N is N1 / N2}.
expression(N) --> expression_switch(N,Y), {nn(expression, Y), domain(Y, [0,1,2])}.
expression_switch(N,0) --> term(N).
expression_switch(N,1) --> expression(N1), operator(plus), term(N2), {N is N1 + N2}.
expression_switch(N,2) --> expression(N1), operator(minus), term(N2), {N is N1 - N2}.