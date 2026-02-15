from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Generator:
    """One-body generator E_pq represented by orbital indices (p,q)."""

    p: int
    q: int

    @property
    def dir(self) -> str:
        """Direction type: 'R' if p<q else 'L'."""

        return "R" if int(self.p) < int(self.q) else "L"

    @property
    def tail(self) -> int:
        """Tail/start index = min(p,q)."""

        return min(int(self.p), int(self.q))

    @property
    def head(self) -> int:
        """Head/stop index = max(p,q)."""

        return max(int(self.p), int(self.q))


def overlap_range(gen1: Generator, gen2: Generator) -> range:
    """Inclusive overlap range [max(tail1,tail2), ..., min(head1,head2)].

    Returns an empty range if there is no overlap.
    """

    start = max(gen1.tail, gen2.tail)
    end = min(gen1.head, gen2.head)
    if end < start:
        return range(0, 0)
    return range(start, end + 1)


def mark(gen: Generator, seg: int) -> str:
    """Per-generator boundary marker at overlap segment index `seg`.

    - 'T' = tail/start (underline in thesis tables)
    - 'H' = head/stop (overline in thesis tables)
    - 'N' = none/internal overlap segment
    """

    seg = int(seg)
    if seg == gen.tail:
        return "T"
    if seg == gen.head:
        return "H"
    return "N"


def seg_code(gen1: Generator, gen2: Generator, seg: int) -> str:
    """Return the 4-character seg_code: dir1+mark1+dir2+mark2.

    Examples:
      internal RL      -> RNLN
      underline RL     -> RTLT (both tails)
      overline  RL     -> RHLH (both heads)
      barR underL      -> RHLT (first head, second tail)
    """

    return f"{gen1.dir}{mark(gen1, seg)}{gen2.dir}{mark(gen2, seg)}"

