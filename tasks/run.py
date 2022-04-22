import importlib
import subprocess
from utils.hparams import set_hparams, hparams


def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    task_cls.start()


if __name__ == '__main__':
    try:
        import libtmux

        tmux_session_name = subprocess.run(
            "echo $(tmux list-panes -t \"$TMUX_PANE\" -F '#S' | head -n1)",
            shell=True, check=True, stdout=subprocess.PIPE).stdout
        tmux_session_name = tmux_session_name.decode().strip()
        server = libtmux.Server()
        session = server.find_where({"session_name": tmux_session_name})
        window = session.attached_window
    except Exception as e:
        print('| libtmux load error.')

    try:
        from setproctitle import setproctitle

        # hide the process title
        setproctitle("python train.py")
    except:
        pass
    set_hparams()
    try:
        if hparams['rename_tmux'] and not hparams['infer']:
            window.rename_window('_'.join(hparams['exp_name'].split("_")[:-1]))
    except:
        pass

    run_task()
