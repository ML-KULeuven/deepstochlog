bracket(Y) --> [X], { domain(Y,["(",")"]),  nn(bracket_nn,X, Y)}.

s(0) --> state(_, not_ok), {p(0.5)}.
s(1) --> state(0, ok), {p(0.5)}.

state(s(0), ok) --> bracket(")"), {p(0.125)}.
state(X, not_ok) --> bracket(")"), {X\=s(0), p(0.05)}.
state(_, not_ok) --> bracket("("), {p(0.075)}.
state(X, S) --> bracket("("), state(s(X), S), {p(0.25)}.
state(s(X), S) --> bracket(")"), state(X, S), {p(0.25)}.
state(0, not_ok) --> bracket(")"), state(0, _), {p(0.25)}.