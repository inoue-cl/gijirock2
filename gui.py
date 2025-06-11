import os
from dotenv import load_dotenv
from PySide6 import QtWidgets
from PySide6.QtCore import QLibraryInfo
from main import transcribe


def set_qt_plugin_path():
    if 'QT_PLUGIN_PATH' in os.environ:
        return
    plugin_path = QLibraryInfo.path(QLibraryInfo.PluginsPath)
    if os.path.isdir(plugin_path):
        os.environ['QT_PLUGIN_PATH'] = plugin_path

class TokenDialog(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Transcriber GUI')
        self.layout = QtWidgets.QVBoxLayout(self)

        self.token_edit = QtWidgets.QLineEdit()
        self.token_edit.setPlaceholderText('HuggingFace Token')
        self.model_edit = QtWidgets.QLineEdit('openai/whisper-small')
        self.file_edit = QtWidgets.QLineEdit()
        self.browse_btn = QtWidgets.QPushButton('Browse')
        self.run_btn = QtWidgets.QPushButton('Run')
        self.output = QtWidgets.QTextEdit()

        self.layout.addWidget(self.token_edit)
        self.layout.addWidget(self.model_edit)
        file_layout = QtWidgets.QHBoxLayout()
        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(self.browse_btn)
        self.layout.addLayout(file_layout)
        self.layout.addWidget(self.run_btn)
        self.layout.addWidget(self.output)

        self.browse_btn.clicked.connect(self.select_file)
        self.run_btn.clicked.connect(self.run)

    def select_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Audio')
        if path:
            self.file_edit.setText(path)

    def run(self):
        token = self.token_edit.text().strip()
        if token:
            os.environ['HF_TOKEN'] = token
        load_dotenv()
        result = transcribe(self.file_edit.text(), self.model_edit.text())
        self.output.setPlainText(result)


def main():
    set_qt_plugin_path()
    app = QtWidgets.QApplication([])
    dlg = TokenDialog()
    dlg.show()
    app.exec()

if __name__ == '__main__':
    main()
