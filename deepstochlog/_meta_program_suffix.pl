:- table solve/1.
:- discontiguous skip_predicate/1.
is_builtin(A) :- A \= true, predicate_property(A,built_in).
include_atom(A) :- A =.. [Func | _], include_predicate(Func).



%compound_to_list/2 translates a compound term to a python compatible list, where the first element is either:
% - a functor, meaning that the list represent a predicate of that functor whose arguments are the following elements
% - the term `list`, meaning that the list is a Prolog list, and the following elements are the elements of the list.
compound_to_list(C,L):- var(C), L = ["_"].
compound_to_list(C,L):- is_list(C), maplist(compound_to_list,C,LC), L =["list"|LC].
compound_to_list(C,L):- \+is_list(C), \+ var(C), C =.. [Funct|Args], maplist(compound_to_list,Args,LArgs), atom_string(Funct,FS), L=[FS|LArgs].


% The solve predicate is the actual engine. It is a meta interpreter with tabling.
% It prints clauses that has include_predicate in the head.
solve(true) :- !.
solve((A,B)) :- solve(A), solve(B).
solve(H) :- include_atom(H),
            clause(H,B),
            solve(B),
            write_ground_clause(H,B), fail. %this is a failure-driven loop. It is the only loop-like behaviour that works properly with SWIPL tabling.
solve(H) :- include_atom(H),
            clause(H,B),
            solve(B).
solve(A) :- A \= (_,_), A \= true, \+ include_atom(A), call(A).

% Top-entry point for the solver. It writes elements for queries with variables (e.g. using the "_" variable)
execute_query(Query) :- copy_term(Query, OriginalQuery),
                        term_variables(Query, Vars),
                        (length(Vars, 0) ->  foreach(solve(Query), true);
                                             foreach(solve(Query), write_ground_clause(OriginalQuery, Query))).


% Print a single line of the table using a python compatible syntax
write_ground_clause(Head, Body):-
            comma_list(Body,BL),
            include(include_atom, BL, BLL),
            (BLL=[] -> true; (
                        ConjBody =.. [conj|BLL],
                        compound_to_list(Head,LHead),
                        compound_to_list(ConjBody,LLBody),
                        write_term([LHead,LLBody],[quoted(true)]),
                        nl)).

% The main/0 executes the queries, potentially writing the top queries entries in the tables.
% Then it writes all the entries of the table to the output stream.
main :- maplist(execute_query, [{query}]).

