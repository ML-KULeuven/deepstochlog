digit_dom(Y) :- member(Y,[0,1,2,3,4,5,6,7,8,9]).

nn(number, [X], Y, digit_dom) :: digit(Y) --> [X].
1 :: addition(N) --> digit(N1),
                     digit(N2),
                     {N is N1 + N2}.