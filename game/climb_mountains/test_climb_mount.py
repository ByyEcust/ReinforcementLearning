from game.climb_mountains.climb_mountains import ClimbMountain


def test_climb_mount():
    game = ClimbMountain("mount_cfg.json")
    for _ in range(5):
        game.step(3)
    for _ in range(3):
        game.step(4)
    loc = game.get_state()
    assert(loc == [3, 5]), "Game assertion failed."
