# Troy

Troy is the chess engine Team Trojan Horse entered in the AI Chessathon 2026, the
UK's first chess hackathon, co-hosted by Optiver. This repository is the browser
build: open the page and play it, with no install, no account and no server.

**Play it:** https://aerodart.github.io/troy-chess/

## How it plays

Troy is a negamax alpha-beta search with a tapered piece-square evaluation. The
evaluation is 812 weights covering material, piece-square tables for both game
phases, a bishop pair bonus, passed pawns by rank and king safety, and those
weights were fit by logistic regression on 725,000 positions each labelled with
how its game actually ended. Because the score is linear in its own parameters,
a position reduces to a sparse feature vector and a training pass is one sparse
matrix multiply rather than 725,000 calls into the engine.

The search carries a transposition table, null-move pruning, late move
reductions, a check extension and a quiescence search that keeps looking until
the captures run out, so at depth eight it visits about 170,000 nodes where
plain alpha-beta needs 3.2 million.

## How it runs in a browser

The page loads [Pyodide](https://pyodide.org), which is CPython compiled to
WebAssembly, and runs `agent.py` unchanged inside it. Move generation comes from
python-chess, served here as a wheel because PyPI publishes only an sdist and
micropip installs wheels. The search runs in a web worker so a move that takes
five seconds does not freeze the board.

The competition build gets its speed from a second implementation of the board
and search compiled with numba, which `agent.py` imports inside a `try` block.
There is no LLVM in a browser, so that import fails here and the reference
engine plays instead, exactly as it would on the platform if the compiled import
threw. That path searches a few hundred times fewer positions per second, so
Troy is a good deal weaker in this page than it is on a laptop, reaching roughly
four plies where the compiled build reaches nine. It still beats most casual
players, and it finds mate in one in two milliseconds.

## What is here

`index.html` is the board and the clocks, `engine.worker.js` boots Pyodide and
answers move requests, `agent.py` is the engine as submitted, and
`weights/book.json` is the opening book it plays from. The wheel is
`chess-1.11.2-py3-none-any.whl`, pinned to the version the engine was tested
against. Nothing else is needed, and nothing is sent anywhere.

The rest of the project, which is the compiled engine, the Texel tuning
pipeline, the SPRT test harness and the record of the rated games, lives in a
separate private repository.

## Credits

A production by Advait Bagri and Jonathan Au Yeung, competing as Team Trojan
Horse from Singapore.
