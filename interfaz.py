import sys
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QInputDialog,
    QGridLayout, QLineEdit, QComboBox, QDialog
)
from PyQt6.QtCore import Qt

import matplotlib
matplotlib.use("QtAgg")  # Backend nativo moderno para evitar bugs de pantalla negra
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class DialogoResultado(QDialog):
    """Ventana emergente para mostrar gráficos con Matplotlib dentro de PyQt"""
    def __init__(self, fig, titulo="Resultados", parent=None):
        super().__init__(parent)
        self.setWindowTitle(titulo)
        self.resize(950, 650)
        
        layout = QVBoxLayout(self)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        self.setLayout(layout)


class DialogoConvolucion(QDialog):
    """Ventana para seleccionar el tamaño de máscara y editar sus coeficientes"""
    def __init__(self, img_rgb, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de Máscara de Convolución")
        self.resize(420, 380)
        self.img_rgb = img_rgb
        self.entradas = []

        layout = QVBoxLayout(self)

        # Selección de tamaño
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Tamaño de máscara:"))
        self.combo_tam = QComboBox()
        self.combo_tam.addItems(["2x2", "3x3", "5x5"])
        self.combo_tam.setCurrentText("3x3")
        self.combo_tam.currentTextChanged.connect(self.actualizar_matriz)
        h_layout.addWidget(self.combo_tam)
        layout.addLayout(h_layout)

        # Contenedor de la matriz
        self.grid_matriz = QGridLayout()
        layout.addLayout(self.grid_matriz)

        # Botón aplicar
        self.btn_aplicar = QPushButton("Aplicar Convolución")
        self.btn_aplicar.clicked.connect(self.aplicar_filtro)
        layout.addWidget(self.btn_aplicar)

        self.actualizar_matriz("3x3")

    def actualizar_matriz(self, tam_str):
        # Limpiar matriz previa
        for i in reversed(range(self.grid_matriz.count())):
            widget = self.grid_matriz.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        self.entradas.clear()

        n = int(tam_str.split("x")[0])
        val_default = f"{round(1.0 / (n * n), 3)}"

        for r in range(n):
            fila = []
            for c in range(n):
                line_edit = QLineEdit(val_default)
                line_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.grid_matriz.addWidget(line_edit, r, c)
                fila.append(line_edit)
            self.entradas.append(fila)

    def aplicar_filtro(self):
        n = int(self.combo_tam.currentText().split("x")[0])
        kernel = np.zeros((n, n), dtype=np.float32)

        try:
            for r in range(n):
                for c in range(n):
                    kernel[r, c] = float(self.entradas[r][c].text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Ingresa únicamente valores numéricos en el kernel.")
            return

        # Aplicar convolución espacial 2D
        img_filtrada = cv2.filter2D(self.img_rgb, -1, kernel)

        # Crear figura
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        axes[0].imshow(self.img_rgb)
        axes[0].set_title("Imagen Original")
        axes[0].axis("off")

        axes[1].imshow(img_filtrada)
        axes[1].set_title(f"Resultado Convolución ({n}x{n})")
        axes[1].axis("off")

        fig.tight_layout()

        # Mostrar resultado
        res_dialog = DialogoResultado(fig, f"Resultado Convolución ({n}x{n})", self)
        res_dialog.exec()


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesamiento Digital de Imágenes")
        self.resize(380, 200)

        widget_central = QWidget()
        layout = QVBoxLayout(widget_central)

        lbl = QLabel("Seleccione una operación:")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(lbl)

        btn_ecualizacion = QPushButton("Ecualización")
        btn_ecualizacion.setFixedHeight(40)
        btn_ecualizacion.clicked.connect(self.ejecutar_ecualizacion)
        layout.addWidget(btn_ecualizacion)

        btn_convolucion = QPushButton("Convolución")
        btn_convolucion.setFixedHeight(40)
        btn_convolucion.clicked.connect(self.ejecutar_convolucion)
        layout.addWidget(btn_convolucion)

        self.setCentralWidget(widget_central)

    def seleccionar_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar Imagen",
            "",
            "Archivos de Imagen (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        return ruta

    def ejecutar_ecualizacion(self):
        ruta = self.seleccionar_imagen()
        if not ruta:
            return

        img_orig = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img_orig is None:
            QMessageBox.critical(self, "Error", "No se pudo cargar la imagen.")
            return

        # Función empleada: cv2.equalizeHist()
        img_eq = cv2.equalizeHist(img_orig)

        fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
        fig.suptitle(
            "Función utilizada: cv2.equalizeHist()\n(Mapeo por Función de Distribución Acumulada - CDF normalizada)",
            fontsize=11,
            fontweight="bold",
            color="#0f4c81"
        )

        # 1. Original
        axes[0, 0].imshow(img_orig, cmap="gray", vmin=0, vmax=255)
        axes[0, 0].set_title("Imagen Original")
        axes[0, 0].axis("off")

        # 2. Histograma Original
        axes[0, 1].hist(img_orig.ravel(), bins=256, range=[0, 256], color="#2c3e50")
        axes[0, 1].set_title("Histograma Original")
        axes[0, 1].set_xlim([0, 256])
        axes[0, 1].grid(True, linestyle="--", alpha=0.6)

        # 3. Ecualizada
        axes[1, 0].imshow(img_eq, cmap="gray", vmin=0, vmax=255)
        axes[1, 0].set_title("Imagen Ecualizada")
        axes[1, 0].axis("off")

        # 4. Histograma Ecualizado
        axes[1, 1].hist(img_eq.ravel(), bins=256, range=[0, 256], color="#27ae60")
        axes[1, 1].set_title("Histograma Ecualizado")
        axes[1, 1].set_xlim([0, 256])
        axes[1, 1].grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout()

        dialogo = DialogoResultado(fig, "Resultado de Ecualización de Histograma", self)
        dialogo.exec()

    def ejecutar_convolucion(self):
        ruta = self.seleccionar_imagen()
        if not ruta:
            return

        img = cv2.imread(ruta)
        if img is None:
            QMessageBox.critical(self, "Error", "No se pudo cargar la imagen.")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dialogo = DialogoConvolucion(img_rgb, self)
        dialogo.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())
