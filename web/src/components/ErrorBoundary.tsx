import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode };
type State = { error: Error | null };

/** Stops a single component error from blanking the whole app (black screen). */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("DubStudio UI error:", error, info);
  }

  reset = () => this.setState({ error: null });

  render() {
    if (this.state.error) {
      return (
        <main>
          <section className="card">
            <h2>Something broke in the UI</h2>
            <p className="muted">
              The page hit an error but the server is fine — your job keeps running. Reload to
              reconnect.
            </p>
            <pre className="err" style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>
              {String(this.state.error?.message || this.state.error)}
            </pre>
            <div className="downloads">
              <button onClick={() => location.reload()}>Reload</button>
              <button className="btn ghost" onClick={this.reset}>
                Dismiss
              </button>
            </div>
          </section>
        </main>
      );
    }
    return this.props.children;
  }
}
