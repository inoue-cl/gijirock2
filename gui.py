import os
from dotenv import load_dotenv
from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import QLibraryInfo
from main import transcribe, format_segments


def set_qt_plugin_path():
    if 'QT_PLUGIN_PATH' in os.environ:
        return
    plugin_path = QLibraryInfo.path(QLibraryInfo.PluginsPath)
    if os.path.isdir(plugin_path):
        os.environ['QT_PLUGIN_PATH'] = plugin_path


class TranscribeThread(QtCore.QThread):
    finished = QtCore.Signal(str)

    def __init__(self, path: str, model: str):
        super().__init__()
        self.path = path
        self.model = model

    def run(self):
        try:
            segments = transcribe(self.path, self.model)
            result = format_segments(segments)
        except Exception as e:
            result = f"Error: {e}"
        self.finished.emit(result)

class TranscriberGUI(QtWidgets.QWidget):
    MODELS = [
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v3",
        "distil-whisper/distil-medium.en",
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Transcriber GUI')
        self.layout = QtWidgets.QVBoxLayout(self)

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(self.MODELS)

        self.file_edit = QtWidgets.QLineEdit()
        self.browse_btn = QtWidgets.QPushButton('Browse')
        self.run_btn = QtWidgets.QPushButton('Run')
        self.output = QtWidgets.QTextEdit()
        self.output.setReadOnly(True)

        self.layout.addWidget(self.model_combo)
        file_layout = QtWidgets.QHBoxLayout()
        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(self.browse_btn)
        self.layout.addLayout(file_layout)
        self.layout.addWidget(self.run_btn)
        self.layout.addWidget(self.output)

        self.browse_btn.clicked.connect(self.select_file)
        self.run_btn.clicked.connect(self.start_transcription)

        load_dotenv()
        self.thread = None

    def select_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Audio')
        if path:
            self.file_edit.setText(path)

    def start_transcription(self):
        if not self.file_edit.text():
            return
        self.run_btn.setEnabled(False)
        model = self.model_combo.currentText()
        path = self.file_edit.text()
        self.output.setPlainText('Running...')

        self.thread = TranscribeThread(path, model)
        self.thread.finished.connect(self.display_result)
        self.thread.start()

    def display_result(self, text):
        self.output.setPlainText(text)
        self.run_btn.setEnabled(True)


def main():
    set_qt_plugin_path()
    app = QtWidgets.QApplication([])
    dlg = TranscriberGUI()
    dlg.show()
    app.exec()

if __name__ == '__main__':
    main()
