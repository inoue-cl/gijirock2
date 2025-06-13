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

    def __init__(self, path: str, model: str, params: dict):
        super().__init__()
        self.path = path
        self.model = model
        self.params = params

    def run(self):
        try:
            segments = transcribe(self.path, self.model,
                                  threshold=self.params.get('threshold', 0.5),
                                  min_cluster_size=self.params.get('cluster_size', 15),
                                  min_on=self.params.get('min_on', 0.5),
                                  min_off=self.params.get('min_off', 0.5))
            result = format_segments(segments)
        except Exception as e:
            result = f"Error: {e}"
        self.finished.emit(result)


class FileDropEdit(QtWidgets.QLineEdit):
    """QLineEdit that accepts drag-and-drop of audio files."""

    AUDIO_EXT = {'.wav', '.mp3', '.m4a'}

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().endswith(tuple(self.AUDIO_EXT)):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if path:
                self.setText(path)

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

        self.file_edit = FileDropEdit()
        self.file_edit.setPlaceholderText('Drop audio file or browse')
        self.browse_btn = QtWidgets.QPushButton('Browse')
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(0.5)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(['Default', 'Noisy', 'Clean'])
        self.cluster_spin = QtWidgets.QSpinBox()
        self.cluster_spin.setRange(1, 50)
        self.cluster_spin.setValue(15)
        self.min_on_spin = QtWidgets.QDoubleSpinBox()
        self.min_on_spin.setRange(0.0, 2.0)
        self.min_on_spin.setSingleStep(0.1)
        self.min_on_spin.setValue(0.5)
        self.min_off_spin = QtWidgets.QDoubleSpinBox()
        self.min_off_spin.setRange(0.0, 2.0)
        self.min_off_spin.setSingleStep(0.1)
        self.min_off_spin.setValue(0.5)

        self.run_btn = QtWidgets.QPushButton('Run')
        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        self.output = QtWidgets.QTextEdit()
        self.output.setReadOnly(True)
        self.save_btn = QtWidgets.QPushButton('Save')
        self.save_btn.setEnabled(False)

        self.apply_preset('Default')

        self.layout.addWidget(self.model_combo)
        self.layout.addWidget(self.preset_combo)
        file_layout = QtWidgets.QHBoxLayout()
        file_layout.addWidget(self.file_edit)
        file_layout.addWidget(self.browse_btn)
        self.layout.addLayout(file_layout)
        param_layout = QtWidgets.QFormLayout()
        param_layout.addRow('Threshold', self.threshold_spin)
        param_layout.addRow('Cluster Size', self.cluster_spin)
        param_layout.addRow('Min On', self.min_on_spin)
        param_layout.addRow('Min Off', self.min_off_spin)
        self.layout.addLayout(param_layout)
        self.layout.addWidget(self.run_btn)
        self.layout.addWidget(self.progress)
        self.layout.addWidget(self.output)
        self.layout.addWidget(self.save_btn)

        self.browse_btn.clicked.connect(self.select_file)
        self.run_btn.clicked.connect(self.start_transcription)
        self.save_btn.clicked.connect(self.save_text)
        self.preset_combo.currentTextChanged.connect(self.apply_preset)

        load_dotenv()
        self.thread = None
        self.current_text = ''

    def apply_preset(self, name: str):
        if name == 'Noisy':
            self.threshold_spin.setValue(0.3)
            self.min_on_spin.setValue(0.3)
            self.min_off_spin.setValue(0.3)
        elif name == 'Clean':
            self.threshold_spin.setValue(0.6)
            self.min_on_spin.setValue(0.5)
            self.min_off_spin.setValue(0.5)
        else:
            self.threshold_spin.setValue(0.5)
            self.min_on_spin.setValue(0.5)
            self.min_off_spin.setValue(0.5)

    def select_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select Audio')
        if path:
            self.file_edit.setText(path)

    def start_transcription(self):
        if not self.file_edit.text():
            return
        self.run_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        model = self.model_combo.currentText()
        path = self.file_edit.text()
        self.output.setPlainText('Running...')

        params = {
            'threshold': self.threshold_spin.value(),
            'cluster_size': self.cluster_spin.value(),
            'min_on': self.min_on_spin.value(),
            'min_off': self.min_off_spin.value(),
        }
        self.thread = TranscribeThread(path, model, params)
        self.thread.finished.connect(self.display_result)
        self.thread.start()

    def display_result(self, text):
        self.output.setPlainText(text)
        self.current_text = text
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)

    def save_text(self):
        if not self.current_text:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Text', filter='Text Files (*.txt)')
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.current_text)


def main():
    set_qt_plugin_path()
    app = QtWidgets.QApplication([])
    dlg = TranscriberGUI()
    dlg.show()
    app.exec()

if __name__ == '__main__':
    main()
