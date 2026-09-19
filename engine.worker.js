/* Troy's engine, running inside the visitor's browser.
 *
 * Pyodide is CPython compiled to WebAssembly, so this loads agent.py unchanged:
 * the same negamax search, tapered piece-square evaluation and opening book that
 * played the rated games. What is missing is the numba layer, because there is no
 * LLVM in a browser, so the guarded import at the top of agent.py fails and the
 * reference engine plays instead. That is the same fallback the competition build
 * uses when the compiled import throws, and it is a few hundred times slower, so
 * expect roughly depth four rather than depth nine.
 *
 * Everything runs here rather than on the page's thread so a search that takes
 * five seconds does not freeze the board underneath it.
 */

const PYODIDE_VERSION = "0.28.0";
const CHESS_WHEEL = "chess-1.11.2-py3-none-any.whl";

importScripts(`https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/pyodide.js`);

let pyodide = null;

function status(text) {
  postMessage({ type: "status", text: text });
}

async function boot() {
  status("Starting Python");
  pyodide = await loadPyodide({
    indexURL: `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`,
    // agent.py reports its environment on stderr as it imports. Useful in the
    // console, but it is not an error and must not look like one.
    stderr: (line) => console.log("[troy]", line),
  });

  // python-chess publishes only an sdist to PyPI and micropip installs wheels,
  // so the wheel is built from the same version the engine was tested against
  // and served from this origin.
  status("Installing python-chess");
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(new URL(CHESS_WHEEL, location.href).href);

  status("Loading Troy");
  const [agentSource, book] = await Promise.all([
    fetch("agent.py").then((r) => r.text()),
    fetch("weights/book.json").then((r) => r.text()),
  ]);
  // agent.py finds its book at `Path(__file__).parent / "weights" / "book.json"`,
  // so the two have to sit in that shape inside Pyodide's virtual filesystem.
  pyodide.FS.mkdirTree("/troy/weights");
  pyodide.FS.writeFile("/troy/agent.py", agentSource);
  pyodide.FS.writeFile("/troy/weights/book.json", book);

  await pyodide.runPythonAsync(`
import os, sys
os.environ["AC_SKIP_NUMBA_PROBE"] = "1"   # the probe imports numba, which is not here
sys.path.insert(0, "/troy")
import agent
`);

  postMessage({
    type: "ready",
    bookEntries: pyodide.runPython("len(agent._OPENING_BOOK)"),
    compiled: pyodide.runPython("bool(agent._has_compiled_search)"),
  });
}

onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === "move") {
      pyodide.globals.set("_fen", message.fen);
      pyodide.globals.set("_ms", message.timeLeftMs);
      const uci = await pyodide.runPythonAsync("agent.get_move(_fen, int(_ms))");
      postMessage({ type: "move", uci: uci, id: message.id });
    } else if (message.type === "reset") {
      // The repetition guard reads a running tally of positions seen this game,
      // so a new game has to start from an empty one or it will refuse moves.
      await pyodide.runPythonAsync(`
agent._game_position_counts.clear()
agent._has_logged_start_position = False
`);
      postMessage({ type: "reset-done" });
    }
  } catch (error) {
    postMessage({ type: "error", text: String(error), id: message.id });
  }
};

boot().catch((error) => {
  postMessage({ type: "error", text: String(error), fatal: true });
});
