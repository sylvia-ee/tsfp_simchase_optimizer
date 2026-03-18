def main():
    print("Hello from tsfp-influence-island!")


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        from streamlit.web.cli import main
        import sys
        sys.argv = ['streamlit', 'run', 'gui.py', '--server.address=0.0.0.0']
        main()
