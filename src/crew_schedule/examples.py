"""Handcrafted construction and micro benchmark instances."""

from __future__ import annotations

from .models import ProjectInstance, Task


def toy_project() -> ProjectInstance:
    """Four unit tasks in a fork/join graph with one crew; optimum is four."""

    tasks = (
        Task("prep", "Site preparation", 1),
        Task("foundation", "Foundation", 1),
        Task("utilities", "Utility rough-in", 1),
        Task("inspection", "Inspection", 1),
    )
    precedence = (
        ("prep", "foundation"),
        ("prep", "utilities"),
        ("foundation", "inspection"),
        ("utilities", "inspection"),
    )
    return ProjectInstance(tasks, precedence, crew_limit=1, name="four_task_toy")


def construction_project() -> ProjectInstance:
    """Ten-task construction project with three interchangeable crews."""

    tasks = (
        Task("site_prep", "Site preparation", 2),
        Task("foundation", "Foundation", 3),
        Task("framing", "Framing", 4),
        Task("roofing", "Roofing", 3),
        Task("plumbing", "Plumbing rough-in", 2),
        Task("electrical", "Electrical rough-in", 2),
        Task("insulation", "Insulation", 2),
        Task("drywall", "Drywall", 3),
        Task("interior", "Interior finish", 2),
        Task("inspection", "Final inspection", 1),
    )
    precedence = (
        ("site_prep", "foundation"),
        ("foundation", "framing"),
        ("framing", "roofing"),
        ("framing", "plumbing"),
        ("framing", "electrical"),
        ("roofing", "insulation"),
        ("plumbing", "insulation"),
        ("electrical", "insulation"),
        ("insulation", "drywall"),
        ("drywall", "interior"),
        ("roofing", "inspection"),
        ("interior", "inspection"),
    )
    return ProjectInstance(tasks, precedence, crew_limit=3, name="construction_10")


def micro_projects() -> tuple[ProjectInstance, ...]:
    """QAOA-sized projects sharing the same time-indexed formulation."""

    two = ProjectInstance(
        (Task("a", "Excavate A", 1), Task("b", "Excavate B", 1)),
        (),
        crew_limit=1,
        name="micro_2",
    )
    three = ProjectInstance(
        (
            Task("a", "Preparation", 1),
            Task("b", "Work package B", 1),
            Task("c", "Work package C", 1),
        ),
        (("a", "b"), ("a", "c")),
        crew_limit=1,
        name="micro_3",
    )
    return two, three, toy_project()

