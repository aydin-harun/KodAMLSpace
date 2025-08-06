import win32serviceutil
import win32service
import win32event
import subprocess
import os
import logging
from pathlib import Path
import time
import signal
import threading

class MyApiService(win32serviceutil.ServiceFramework):
    _svc_name_ = "KodAMLSpaceService"
    _svc_display_name_ = "KodA ML Space Windows Service"
    _exe_path = r"C:\Users\aydin\Documents\PythonProjects\KodAMLSpace\dist\KodAMLSpace\KodAMLSpace.exe"

    def __init__(self, args):
        super().__init__(args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.process = None
        self.thread = None

        # ✅ loglama ayarı
        self.logger = self._setup_logger()

        self.logger.info("Service initialized")

    def _setup_logger(self):
        log_dir = Path(os.path.dirname(self._exe_path)) / "logs"
        log_dir.mkdir(exist_ok=True)

        logger = logging.getLogger(self._svc_name_)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(log_dir / "service.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def SvcStop(self):
        self.logger.info("Service stop requested")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)

        if self.process and self.process.poll() is None:
            self.logger.info("Terminating child process gracefully")
            try:
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(2)
                if self.process.poll() is None:
                    self.process.terminate()
            except Exception as e:
                self.logger.error(f"Error terminating child process: {e}")

    def SvcDoRun(self):
        self.logger.info("Service starting...")
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)

        self.thread = threading.Thread(target=self.main)
        self.thread.start()

    def main(self):
        try:
            self.logger.info(f"Starting child process: {self._exe_path}")
            self.process = subprocess.Popen(
                self._exe_path,
                cwd=os.path.dirname(self._exe_path),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )

            win32event.WaitForSingleObject(self.hWaitStop, win32event.INFINITE)
            self.logger.info("Stop event detected")
        except Exception as e:
            self.logger.exception(f"Error in service run: {e}")
        finally:
            if self.process and self.process.poll() is None:
                self.logger.info("Cleaning up child process")
                self.process.terminate()
            self.logger.info("Service stopped.")

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(MyApiService)
