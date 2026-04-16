from q1 import run as run_q1
from q2 import run as run_q2
from q3 import run as run_q3
from q4 import run as run_q4
from q5 import run as run_q5
from q6 import run as run_q6
from q7 import run as run_q7
from q8 import run as run_q8
from q9 import run as run_q9
from q10 import run as run_q10
from q11 import run as run_q11
from q12 import run as run_q12


def main() -> None:
    for runner in [
        run_q1,
        run_q2,
        run_q3,
        run_q4,
        run_q5,
        run_q6,
        run_q7,
        run_q8,
        run_q9,
        run_q10,
        run_q11,
        run_q12,
    ]:
        runner()
    print("All questions completed.")


if __name__ == "__main__":
    main()
