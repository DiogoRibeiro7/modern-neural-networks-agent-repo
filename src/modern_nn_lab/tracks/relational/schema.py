"""Relational schema: tables, typed columns, foreign keys, and time columns.

The track prompt asks for a representation of *entities, rows, columns, foreign-key links,
timestamps, and task context* that does not flatten every table into one feature matrix.
This module is where that representation is declared, and it is deliberately explicit: a
column knows whether it is numeric, categorical, a key, or a timestamp, and a foreign key
knows which column of which table it points at.

The distinction that matters downstream is between a **time column** and any other column.
A row's time column is what makes it eligible or ineligible for a prediction made at some
timestamp, and :mod:`modern_nn_lab.tracks.relational.sampler` refuses to build a
neighbourhood from a table whose time column it cannot find. Tables without a time column
are *static* — a product's price is not an event — and are declared as such rather than
inferred, because silently treating an event table as static is precisely how temporal
leakage enters a relational pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

ColumnKind = Literal["numeric", "categorical", "primary_key", "foreign_key", "time"]
"""What a column is. Determines how it is encoded and whether it gates on time."""


@dataclass(frozen=True, slots=True)
class Column:
    """One typed column of a table.

    Attributes:
        name: Column name, unique within its table.
        kind: What the column is.
        cardinality: Number of distinct values, for categorical columns only.
    """

    name: str
    kind: ColumnKind
    cardinality: int = 0

    def __post_init__(self) -> None:
        """Validate the column.

        Raises:
            ValueError: If a categorical column has no cardinality, or a non-categorical
                column declares one.
        """

        if self.kind == "categorical" and self.cardinality < 2:
            raise ValueError(f"categorical column {self.name!r} needs cardinality >= 2")
        if self.kind != "categorical" and self.cardinality:
            raise ValueError(f"only categorical columns carry a cardinality; got {self.name!r}")


@dataclass(frozen=True, slots=True)
class ForeignKey:
    """A directed link from one table's column to another table's primary key.

    Attributes:
        table: Table holding the referencing column.
        column: Referencing column name.
        references: Table being referenced.
    """

    table: str
    column: str
    references: str


@dataclass(frozen=True, slots=True)
class Table:
    """A table: typed columns plus the row data itself.

    Rows are held as a dict of column name to a one-dimensional tensor, all of the same
    length. Keeping columns separate rather than as one matrix is what lets a column keep
    its type — which is the whole point of the representation.

    Attributes:
        name: Table name.
        columns: Typed columns.
        data: Column name to values, each of shape ``(n_rows,)``.
    """

    name: str
    columns: tuple[Column, ...]
    data: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the table.

        Raises:
            ValueError: If columns are duplicated, if data does not match the declared
                columns, or if column lengths disagree.
        """

        names = [column.name for column in self.columns]
        if len(set(names)) != len(names):
            raise ValueError(f"table {self.name!r} has duplicate column names")
        if not self.data:
            return

        missing = set(names) - set(self.data)
        extra = set(self.data) - set(names)
        if missing or extra:
            raise ValueError(
                f"table {self.name!r}: data does not match columns "
                f"(missing {sorted(missing)}, undeclared {sorted(extra)})"
            )
        lengths = {int(values.shape[0]) for values in self.data.values()}
        if len(lengths) > 1:
            raise ValueError(f"table {self.name!r}: columns have differing lengths {lengths}")

    @property
    def n_rows(self) -> int:
        """Number of rows."""

        if not self.data:
            return 0
        return int(next(iter(self.data.values())).shape[0])

    def column(self, name: str) -> Column:
        """Return a column by name.

        Args:
            name: Column name.

        Returns:
            The column.

        Raises:
            KeyError: If the table has no such column.
        """

        for candidate in self.columns:
            if candidate.name == name:
                return candidate
        raise KeyError(f"table {self.name!r} has no column {name!r}")

    def column_of_kind(self, kind: ColumnKind) -> Column | None:
        """Return the first column of a kind, or ``None``.

        Args:
            kind: Column kind to look for.

        Returns:
            The column, or ``None`` when the table has none.
        """

        for candidate in self.columns:
            if candidate.kind == kind:
                return candidate
        return None

    @property
    def time_column(self) -> Column | None:
        """The table's time column, or ``None`` for a static table."""

        return self.column_of_kind("time")

    @property
    def is_static(self) -> bool:
        """Whether the table has no time column.

        A static table's rows are facts, not events: they were always true and cannot leak
        information from after a prediction timestamp.
        """

        return self.time_column is None

    def numeric_columns(self) -> tuple[Column, ...]:
        """Columns encoded as real values."""

        return tuple(column for column in self.columns if column.kind == "numeric")

    def categorical_columns(self) -> tuple[Column, ...]:
        """Columns encoded as categories."""

        return tuple(column for column in self.columns if column.kind == "categorical")


@dataclass(frozen=True, slots=True)
class Database:
    """A set of tables and the foreign keys between them.

    Attributes:
        tables: Table name to table.
        foreign_keys: Every declared link.
    """

    tables: dict[str, Table]
    foreign_keys: tuple[ForeignKey, ...]

    def __post_init__(self) -> None:
        """Validate the database.

        Raises:
            ValueError: If a foreign key names a table or column that does not exist, or
                points at a table with no primary key.
        """

        for key in self.foreign_keys:
            if key.table not in self.tables:
                raise ValueError(f"foreign key from unknown table {key.table!r}")
            if key.references not in self.tables:
                raise ValueError(f"foreign key to unknown table {key.references!r}")
            self.tables[key.table].column(key.column)
            if self.tables[key.references].column_of_kind("primary_key") is None:
                raise ValueError(f"table {key.references!r} has no primary key to reference")

    def table(self, name: str) -> Table:
        """Return a table by name.

        Args:
            name: Table name.

        Returns:
            The table.

        Raises:
            KeyError: If no such table exists.
        """

        if name not in self.tables:
            raise KeyError(f"unknown table {name!r}; available: {sorted(self.tables)}")
        return self.tables[name]

    def children_of(self, table: str) -> tuple[ForeignKey, ...]:
        """Foreign keys pointing *at* a table, i.e. its child relations.

        Args:
            table: Referenced table name.

        Returns:
            Every foreign key whose ``references`` is ``table``.
        """

        return tuple(key for key in self.foreign_keys if key.references == table)

    def parent_of(self, table: str, column: str) -> str | None:
        """The table a foreign-key column points at.

        Args:
            table: Referencing table.
            column: Referencing column.

        Returns:
            The referenced table name, or ``None`` when the column is not a foreign key.
        """

        for key in self.foreign_keys:
            if key.table == table and key.column == column:
                return key.references
        return None

    def row_counts(self) -> dict[str, int]:
        """Rows per table, for the record's configuration snapshot."""

        return {name: table.n_rows for name, table in sorted(self.tables.items())}


def index_by_key(values: Tensor) -> dict[int, list[int]]:
    """Build a lookup from key value to the row positions holding it.

    Args:
        values: Shape ``(n_rows,)`` integer key column.

    Returns:
        Mapping from key value to row positions, in ascending row order.
    """

    lookup: dict[int, list[int]] = {}
    for position, value in enumerate(values.tolist()):
        lookup.setdefault(int(value), []).append(position)
    return lookup


def as_long(values: Tensor) -> Tensor:
    """Return an integer view of a key or index column.

    Args:
        values: Any numeric tensor.

    Returns:
        The same values as ``torch.long``.
    """

    return values.to(torch.long)
