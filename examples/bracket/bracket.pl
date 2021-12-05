brackets_dom(X) :- member(X, ["(",")"]).
nn(bracket_nn, [X], Y, brackets_dom) :: bracket(Y) --> [X].

t(_) :: s --> s, s.
t(_) :: s --> bracket("("), s, bracket(")").
t(_) :: s --> bracket("("), bracket(")").