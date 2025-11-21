import argparse

from src.process import record_run


def get_args():
    parser = argparse.ArgumentParser("Record rollouts from a trained PPO Super Mario Bros agent.")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument(
        "--action-type",
        type=str,
        default="custom",
        choices=["custom", "simple", "right", "complex"],
    )
    parser.add_argument("--model-path", type=str, default="trained_models/ppo_super_mario_bros_1_1")
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--recordings-root", type=str, default="recordings")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument(
        "--quality",
        type=str,
        default="high",
        choices=["low", "medium", "high"],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=25.0,
        help="Action sampling temperature; 0 for greedy argmax",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = get_args()
    for i in range(2):
        record_run(
            opt=opt,
            model_path=opt.model_path,
            num_episodes=opt.num_episodes,
            recordings_root=opt.recordings_root,
            frame_skip=opt.frame_skip,
            quality=opt.quality,
            temperature=opt.temperature-i,
        )
