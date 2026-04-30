# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python data science study project organized as a progressive curriculum covering statistical learning, optimization, and advanced data analysis. Uses `uv` for package management.

## Commands

```bash
# Install dependencies
uv sync

# Run a specific lesson script
uv run python Lesson_X/script_name.py

# Run tests (Lesson 6 has test files)
uv run python Lesson_6/CP_SAT_Solver_test.py
```

## Architecture

**Lesson-based structure** progressing from foundational to advanced topics:

- **Lesson_1_2**: Web scraping (requests, BeautifulSoup), HTTP methods, data export
- **Lesson_3**: Polynomial/exponential regression, anomaly detection and restoration
- **Lesson_4**: Alpha-Beta filtering, OpenCV object tracking (MeanShift, CamShift)
- **Lesson_5**: Non-linear regression, bisection method, SymPy symbolic math
- **Lesson_6**: CP-SAT Solver (Google OR-Tools), job scheduling optimization
- **Lesson_7**: Multi-criteria decision making, scoring/evaluation models
- **Lesson_8**: OLAP 3D visualization, text mining, voice recognition

**Standard lesson file pattern**:
1. Data parsing (typically from `Oschadbank (USD).xls`)
2. Statistical characterization
3. Algorithm implementation
4. Matplotlib visualization
5. Output generation (plots, CSV/JSON/Excel)

## Code Style

- Function-based modular design (no classes in core lessons)
- No type hints unless Pydantic is used
- Comments only for complex algorithm sections
- Ukrainian language in documentation headers

## Key Dependencies

- NumPy/SciPy: Polynomial fitting, curve fitting
- Pandas: Data frame operations, Excel I/O
- Matplotlib/Seaborn: Visualization
- OpenCV: Object tracking
- SymPy: Symbolic mathematics
- OR-Tools: Constraint programming
- Requests/BeautifulSoup: Web scraping
