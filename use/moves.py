from mcpy.moves import DeletionMove, InsertionMove, DisplacementMove
from mcpy.moves.move_selector import MoveSelector


species = ['Cu']

move_list = [[25, 25, 50],
             [DeletionMove(species=species, seed=12),
              InsertionMove(species=species, seed=13),
              DisplacementMove(species=species, seed=14)]]

move_selector = MoveSelector(*move_list)
for i in range(100):
    print(move_selector.get_operator())
