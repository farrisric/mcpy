from mcpy.moves import DeletionMove, InsertionMove, DisplacementMove
from mcpy.moves.move_selector import MoveSelector


species = ['Cu']

insert_move = InsertionMove(species=species,
                            operating_box=operating_box,
                            z_shift=z_shift,
                            seed=random_seed+3)

deletion_move = DeletionMove(species=species,
                            operating_box=operating_box,
                            z_shift=z_shift,
                            seed=random_seed+4)

displace_move = DisplacementMove(species=species,
                            seed=_random_seed+5,
                            constraints=atoms.constraints,
                            max_displacement=max_displacement)

move_list = [[25, 25, 50],
             [DeletionMove(species=species, seed=12),
              InsertionMove(species=species, seed=13),
              DisplacementMove(species=species, seed=14)]]

move_selector = MoveSelector(*move_list)
for i in range(100):
    print(move_selector.get_operator())
