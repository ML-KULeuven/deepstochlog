digit(Y) :- member(Y,[0,1,2,3,4,5,6,7,8,9]).
nn(number, [X], Y, digit) :: is_number(Y) --> [X].
addition(N) --> is_number(N1),
                is_number(N2),
                {N is N1 + N2}.
multi_addition(N, 1) --> addition(N).
multi_addition(N, L) --> {L > 1, L2 is L - 1},
                         addition(N1),
                         multi_addition(N2, L2),
                         {N is N1*(10**L2) + N2}.