#include "mainwindow.h"
#include <QDateTime>
#include <QDebug>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QSerialPortInfo>
#include <QVBoxLayout>

// ============================================================================
//  Design Tokens
// ============================================================================
namespace Styles {
const QString BG_DARK   = "#0A0D14";
const QString CARD_BG   = "#12161F";
const QString ACCENT    = "#58A6FF";
const QString SUCCESS   = "#3FB950";
const QString DANGER    = "#FF4C4C";
const QString CYAN      = "#39D0D8";
const QString TEXT_MAIN = "#C9D1D9";
const QString TEXT_DIM  = "#6E7681";
const QString BORDER    = "#21262D";
const QString PURPLE    = "#D2A8FF";
const QString AMBER     = "#F0A737";
const QString CARD_GLOW = "1px solid rgba(88,166,255,0.15)";

const QString MAIN_QSS =
    QString(
        /* Window */
        "QMainWindow { background-color: %1; }"

        /* Generic label */
        "QLabel { color: %2; font-family: 'Segoe UI', 'Inter', sans-serif; }"

        /* Cards */
        "QFrame#Card {"
        "  background-color: %3;"
        "  border: 1px solid %4;"
        "  border-radius: 14px;"
        "}"

        /* Default button (blue accent) */
        "QPushButton {"
        "  background-color: %5;"
        "  color: white;"
        "  border-radius: 8px;"
        "  padding: 10px 16px;"
        "  font-weight: 700;"
        "  font-size: 12px;"
        "  letter-spacing: 0.5px;"
        "  border: none;"
        "}"
        "QPushButton:hover { background-color: #79C0FF; }"
        "QPushButton:pressed { background-color: #388BFD; }"

        /* Stop / Terminate button */
        "QPushButton#StopBtn { background-color: %6; }"
        "QPushButton#StopBtn:hover { background-color: #FF7070; }"

        /* Toggle ON — green (SR) */
        "QPushButton#ToggleSR {"
        "  background-color: %7;"
        "  color: #0D1117;"
        "}"
        "QPushButton#ToggleSR:hover { background-color: #56D364; }"

        /* Toggle OFF — dim */
        "QPushButton#ToggleSROff {"
        "  background-color: #21262D;"
        "  color: %8;"
        "  border: 1px solid #30363D;"
        "}"
        "QPushButton#ToggleSROff:hover { background-color: #2D333B; }"

        /* RIFE toggle ON — cyan */
        "QPushButton#ToggleRIFE {"
        "  background-color: %9;"
        "  color: #0D1117;"
        "}"
        "QPushButton#ToggleRIFE:hover { background-color: #5CE8F0; }"

        /* RIFE toggle OFF */
        "QPushButton#ToggleRIFEOff {"
        "  background-color: #21262D;"
        "  color: %8;"
        "  border: 1px solid #30363D;"
        "}"
        "QPushButton#ToggleRIFEOff:hover { background-color: #2D333B; }"

        /* ComboBox */
        "QComboBox {"
        "  background-color: %3;"
        "  color: %2;"
        "  border: 1px solid #30363D;"
        "  border-radius: 8px;"
        "  padding: 7px 12px;"
        "}"
        "QComboBox::drop-down { border: none; }"
        "QComboBox QAbstractItemView {"
        "  background-color: %3;"
        "  color: %2;"
        "  selection-background-color: %5;"
        "}")
        .arg(BG_DARK, TEXT_MAIN, CARD_BG, BORDER,   // 1-4
             ACCENT, DANGER, SUCCESS, TEXT_DIM,       // 5-8
             CYAN);                                   // 9
} // namespace Styles

// ============================================================================
//  InfoCard  —  small telemetry card
// ============================================================================
class InfoCard : public QFrame {
public:
    QLabel *valLbl;
    QLabel *titleLbl;
    InfoCard(QString title, QString unit = "") : QFrame() {
        setObjectName("Card");
        auto layout = new QVBoxLayout(this);
        layout->setContentsMargins(14, 14, 14, 14);
        layout->setSpacing(4);

        titleLbl = new QLabel(title.toUpper());
        titleLbl->setStyleSheet(
            QString("color: %1; font-size: 10px; font-weight: 700; letter-spacing: 1.2px;")
                .arg(Styles::TEXT_DIM));
        layout->addWidget(titleLbl);

        auto row = new QHBoxLayout();
        valLbl = new QLabel("-");
        valLbl->setStyleSheet(
            QString("color: %1; font-size: 20px; font-weight: 700; font-family: 'Consolas';")
                .arg(Styles::PURPLE));
        row->addWidget(valLbl);

        if (!unit.isEmpty()) {
            auto unitLbl = new QLabel(unit);
            unitLbl->setStyleSheet(
                QString("color: %1; font-size: 11px; margin-top: 5px;")
                    .arg(Styles::TEXT_DIM));
            row->addWidget(unitLbl);
        }
        row->addStretch();
        layout->addLayout(row);
    }
    void setValue(const QString &val) { valLbl->setText(val); }
};

// ============================================================================
//  MainWindow
// ============================================================================
MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setupUi();
    setupStyles();
    refreshPorts();
}

MainWindow::~MainWindow() { stopMission(); }

void MainWindow::setupUi() {
    auto central    = new QWidget();
    setCentralWidget(central);
    auto mainLayout = new QVBoxLayout(central);
    mainLayout->setContentsMargins(28, 18, 28, 18);
    mainLayout->setSpacing(18);

    // ── Header ───────────────────────────────────────────────────────────────
    auto header     = new QHBoxLayout();
    auto logoLayout = new QVBoxLayout();
    auto logoRow    = new QHBoxLayout();

    m_logoMain = new QLabel("Long-Range UAV");
    m_logoMain->setStyleSheet(
        QString("color: %1; font-size: 26px; font-weight: 800; letter-spacing: -0.5px;")
            .arg(Styles::ACCENT));
    m_logoSub = new QLabel("Image Transmission Simulation System");
    m_logoSub->setStyleSheet(
        QString("color: %1; font-size: 26px; font-weight: 300;")
            .arg(Styles::TEXT_MAIN));

    logoRow->addWidget(m_logoMain);
    logoRow->addWidget(m_logoSub);
    logoRow->addStretch();
    
    m_langBtn = new QPushButton("🌐 EN / 中");
    m_langBtn->setFixedHeight(36);
    m_langBtn->setStyleSheet(QString("background: transparent; border: 1px solid %1; color: %2;").arg(Styles::BORDER, Styles::TEXT_MAIN));
    connect(m_langBtn, &QPushButton::clicked, this, &MainWindow::toggleLanguage);
    logoRow->addWidget(m_langBtn);

    logoLayout->addLayout(logoRow);

    header->addLayout(logoLayout);
    header->addStretch();

    m_statusLabel = new QLabel("STANDING BY...");
    m_statusLabel->setStyleSheet(
        QString("color: %1; font-weight: 700; font-size: 13px;"
                "background: %2; padding: 10px 18px; border-radius: 8px;"
                "border: 1px solid %3;")
            .arg(Styles::TEXT_MAIN, Styles::CARD_BG, Styles::BORDER));
    header->addWidget(m_statusLabel);
    mainLayout->addLayout(header);

    // Divider line
    auto divider = new QFrame();
    divider->setFrameShape(QFrame::HLine);
    divider->setStyleSheet(QString("background: %1; border: none; max-height: 1px;")
                               .arg(Styles::BORDER));
    mainLayout->addWidget(divider);

    // ── Content area ─────────────────────────────────────────────────────────
    auto content = new QHBoxLayout();
    content->setSpacing(22);

    // ── Left: Video feed ─────────────────────────────────────────────────────
    auto leftSide = new QVBoxLayout();

    m_feedLabel = new QLabel("▶  PRIMARY FEED");
    m_feedLabel->setStyleSheet(
        QString("color: %1; font-size: 10px; font-weight: 700; letter-spacing: 1.5px;")
            .arg(Styles::TEXT_DIM));
    leftSide->addWidget(m_feedLabel);

    auto videoContainer = new QFrame();
    videoContainer->setObjectName("Card");
    videoContainer->setStyleSheet(
        "background: #000000; border-radius: 14px; border: 1px solid #1C2128;");
    auto videoLayout = new QVBoxLayout(videoContainer);
    videoLayout->setContentsMargins(0, 0, 0, 0);
    m_videoLabel = new QLabel();
    m_videoLabel->setAlignment(Qt::AlignCenter);
    m_videoLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    videoContainer->setMinimumSize(820, 620);
    videoLayout->addWidget(m_videoLabel);
    leftSide->addWidget(videoContainer, 1);
    content->addLayout(leftSide, 3);

    // ── Right: Controls & Telemetry ──────────────────────────────────────────
    auto rightSide = new QVBoxLayout();
    rightSide->setSpacing(14);

    // Section label
    m_telLabel = new QLabel("TELEMETRY");
    m_telLabel->setStyleSheet(
        QString("color: %1; font-size: 10px; font-weight: 700; letter-spacing: 1.5px;")
            .arg(Styles::TEXT_DIM));
    rightSide->addWidget(m_telLabel);

    // Telemetry grid: 2×3
    auto grid = new QGridLayout();
    grid->setSpacing(12);

    m_fpsCard     = new InfoCard("Decoded FPS",     "FPS");
    m_rssiCard    = new InfoCard("Link RSSI",       "dBm");
    m_successCard = new InfoCard("Frames OK",       "FRM");
    m_rateCard    = new InfoCard("Success Rate",    "%");
    m_queueCard   = new InfoCard("Display Queue",   "Q");

    m_fpsVal     = m_fpsCard->valLbl;
    m_rssiVal    = m_rssiCard->valLbl;
    m_successVal = m_successCard->valLbl;
    m_rateVal    = m_rateCard->valLbl;
    m_queueVal   = m_queueCard->valLbl;

    grid->addWidget(m_fpsCard,     0, 0);
    grid->addWidget(m_rssiCard,    0, 1);
    grid->addWidget(m_successCard, 1, 0);
    grid->addWidget(m_rateCard,    1, 1);
    grid->addWidget(m_queueCard,   2, 0, 1, 2);
    rightSide->addLayout(grid);

    // ── Control Card ─────────────────────────────────────────────────────────
    m_ctrlLabel = new QLabel("CONTROL");
    m_ctrlLabel->setStyleSheet(
        QString("color: %1; font-size: 10px; font-weight: 700; letter-spacing: 1.5px;")
            .arg(Styles::TEXT_DIM));
    rightSide->addWidget(m_ctrlLabel);

    auto ctrlCard   = new QFrame();
    ctrlCard->setObjectName("Card");
    auto ctrlLayout = new QVBoxLayout(ctrlCard);
    ctrlLayout->setSpacing(12);
    ctrlLayout->setContentsMargins(16, 16, 16, 16);

    // COM port row
    auto cfgRow  = new QHBoxLayout();
    m_portCombo  = new QComboBox();
    auto portIconLabel = new QLabel("📡");
    portIconLabel->setStyleSheet("font-size: 16px;");
    cfgRow->addWidget(portIconLabel);
    cfgRow->addWidget(m_portCombo, 1);
    ctrlLayout->addLayout(cfgRow);

    // Start / Stop button
    m_startBtn = new QPushButton("▶  START LINK");
    m_startBtn->setFixedHeight(46);
    connect(m_startBtn, &QPushButton::clicked, this, &MainWindow::toggleMission);
    ctrlLayout->addWidget(m_startBtn);

    // Divider inside card
    auto innerDiv = new QFrame();
    innerDiv->setFrameShape(QFrame::HLine);
    innerDiv->setStyleSheet(
        QString("background: %1; border: none; max-height: 1px;").arg(Styles::BORDER));
    ctrlLayout->addWidget(innerDiv);

    // AI model label
    m_aiLabel = new QLabel("AI MODELS");
    m_aiLabel->setStyleSheet(
        QString("color: %1; font-size: 10px; font-weight: 700; letter-spacing: 1.2px;")
            .arg(Styles::TEXT_DIM));
    ctrlLayout->addWidget(m_aiLabel);

    // RealESRGAN toggle
    m_srBtn = new QPushButton("◆  REALESRGAN  [ ON ]");
    m_srBtn->setObjectName("ToggleSR");
    m_srBtn->setFixedHeight(44);
    m_srBtn->setToolTip("Toggle RealESRGAN Super-Resolution (4x upscaling)");
    connect(m_srBtn, &QPushButton::clicked, this, &MainWindow::toggleSR);
    ctrlLayout->addWidget(m_srBtn);

    // RIFE toggle
    m_rifeBtn = new QPushButton("◈  RIFE INTERPOLATION  [ ON ]");
    m_rifeBtn->setObjectName("ToggleRIFE");
    m_rifeBtn->setFixedHeight(44);
    m_rifeBtn->setToolTip("Toggle RIFE Frame Interpolation (~12fps → 24fps)");
    connect(m_rifeBtn, &QPushButton::clicked, this, &MainWindow::toggleRIFE);
    ctrlLayout->addWidget(m_rifeBtn);

    // YOLO detected box
    ctrlLayout->addSpacing(4);
    auto yoloBox = new QFrame();
    yoloBox->setStyleSheet(
        QString("background: %1; border: 1px solid %2; border-radius: 8px;")
            .arg(Styles::BG_DARK, Styles::BORDER));
    auto yoloLayout = new QHBoxLayout(yoloBox);
    yoloLayout->setContentsMargins(12, 10, 12, 10);

    m_yoloTitle = new QLabel("YOLO DETECTED:");
    m_yoloTitle->setStyleSheet(
        QString("color: %1; font-size: 11px; font-weight: 700; border: none; background: transparent;")
            .arg(Styles::TEXT_DIM));
            
    m_yoloVal = new QLabel("WAITING...");
    m_yoloVal->setStyleSheet(
        QString("color: %1; font-size: 12px; font-weight: 700; border: none; background: transparent;")
            .arg(Styles::AMBER));

    yoloLayout->addWidget(m_yoloTitle);
    yoloLayout->addWidget(m_yoloVal);
    yoloLayout->addStretch();
    
    ctrlLayout->addWidget(yoloBox);

    rightSide->addWidget(ctrlCard);
    rightSide->addStretch();

    content->addLayout(rightSide, 1);
    mainLayout->addLayout(content);
}

void MainWindow::setupStyles() {
    setStyleSheet(Styles::MAIN_QSS);
    setWindowTitle("TIMWAVE Receiver — v4");
    resize(1280, 820);
}


void MainWindow::toggleLanguage() {
    m_isEnglish = !m_isEnglish;
    updateUITexts();
}

void MainWindow::updateUITexts() {
    if (m_isEnglish) {
        if (m_statusLabel->text().contains("待命中")) m_statusLabel->setText(m_isEnglish ? "STANDING BY..." : "待命中...");
        else if (m_statusLabel->text().contains("連線中")) m_statusLabel->setText("⬤  LINK ACTIVE: " + m_portCombo->currentText());
        else if (m_statusLabel->text().contains("錯誤")) m_statusLabel->setText(m_isEnglish ? "✖  ERROR: PYTHON PATH FAILED" : "✖  錯誤: PYTHON 執行檔失敗");
        
        m_feedLabel->setText("▶  PRIMARY FEED");
        m_telLabel->setText("TELEMETRY");
        
        m_fpsCard->titleLbl->setText("DECODED FPS");
        m_rssiCard->titleLbl->setText("LINK RSSI");
        m_successCard->titleLbl->setText("FRAMES OK");
        m_rateCard->titleLbl->setText("SUCCESS RATE");
        m_queueCard->titleLbl->setText("DISPLAY QUEUE");
        
        m_ctrlLabel->setText("CONTROL");
        m_aiLabel->setText("AI MODELS");
        
        if (m_startBtn->text().contains("連線")) {
            m_startBtn->setText(m_bridge ? "■  TERMINATE LINK" : "▶  START LINK");
        }
        
        m_srBtn->setText(m_srEnabled ? "◆  REALESRGAN  [ ON ]" : "◆  REALESRGAN  [ OFF ]");
        m_rifeBtn->setText(m_rifeEnabled ? "◈  RIFE INTERPOLATION  [ ON ]" : "◈  RIFE INTERPOLATION  [ OFF ]");
        
        m_yoloTitle->setText("YOLO DETECTED:");
        if (m_yoloVal->text() == "等待中...") m_yoloVal->setText("WAITING...");
    } else {
        if (m_statusLabel->text().contains("STANDING BY")) m_statusLabel->setText("待命中...");
        else if (m_statusLabel->text().contains("ACTIVE")) m_statusLabel->setText("⬤  連線中: " + m_portCombo->currentText());
        else if (m_statusLabel->text().contains("ERROR")) m_statusLabel->setText("✖  錯誤: PYTHON 執行檔失敗");
        
        m_feedLabel->setText("▶  主畫面");
        m_telLabel->setText("遙測數據");
        
        m_fpsCard->titleLbl->setText("解碼幀數");
        m_rssiCard->titleLbl->setText("訊號強度");
        m_successCard->titleLbl->setText("正確幀數");
        m_rateCard->titleLbl->setText("成功率");
        m_queueCard->titleLbl->setText("顯示佇列");
        
        m_ctrlLabel->setText("控制面板");
        m_aiLabel->setText("AI 模型");
        
        if (m_startBtn->text().contains("LINK")) {
            m_startBtn->setText(m_bridge ? "■  中斷連線" : "▶  開始連線");
        }
        
        m_srBtn->setText(m_srEnabled ? "◆  REALESRGAN  [ 啟用 ]" : "◆  REALESRGAN  [ 停用 ]");
        m_rifeBtn->setText(m_rifeEnabled ? "◈  RIFE 補幀  [ 啟用 ]" : "◈  RIFE 補幀  [ 停用 ]");
        
        m_yoloTitle->setText("YOLO 偵測：");
        if (m_yoloVal->text() == "WAITING...") m_yoloVal->setText("等待中...");
    }
}

void MainWindow::refreshPorts() {
    m_portCombo->clear();
    QStringList manualPorts = {"COM1", "COM2", "COM3", "COM4", "COM5",
                               "COM6", "COM7", "COM8", "COM9"};
    for (const auto &port : manualPorts)
        m_portCombo->addItem(port, port);
    m_portCombo->setCurrentText("COM3");
}

// ── Toggle Mission ────────────────────────────────────────────────────────────
void MainWindow::toggleMission() {
    if (!m_bridge) {
        QString portName = m_portCombo->currentText();
        if (portName.isEmpty())
            return;

        m_statusLabel->setText(m_isEnglish ? ("⬤  LINK ACTIVE: " + portName) : ("⬤  連線中: " + portName));
        m_statusLabel->setStyleSheet(
            QString("color: %1; font-weight: 700; font-size: 13px;"
                    "background: %2; padding: 10px 18px; border-radius: 8px;"
                    "border: 1px solid %1;")
                .arg(Styles::SUCCESS, Styles::CARD_BG));

        m_startBtn->setText(m_isEnglish ? "■  TERMINATE LINK" : "■  中斷連線");
        m_startBtn->setObjectName("StopBtn");
        m_startBtn->setStyle(m_startBtn->style());

        m_bridge = new QProcess(this);
        QString pythonPath =
            "c:/Users/hoanc/AppData/Local/Programs/Python/Python311/python.exe";
        QString scriptPath = "d:/Qt_project/RX/rx_bridge_v4.py";

        m_bridge->start(pythonPath, QStringList() << scriptPath << portName);

        connect(m_bridge, &QProcess::errorOccurred, this,
                [this](QProcess::ProcessError error) {
                    qDebug() << "Process failed to start:" << error;
                    m_statusLabel->setText(m_isEnglish ? "✖  ERROR: PYTHON PATH FAILED" : "✖  錯誤: PYTHON 執行檔失敗");
                    m_statusLabel->setStyleSheet(
                        QString("color: %1; font-weight: 700; font-size: 13px;"
                                "background: %2; padding: 10px 18px; border-radius: 8px;"
                                "border: 1px solid %1;")
                            .arg(Styles::DANGER, Styles::CARD_BG));
                });

        connect(m_bridge, &QProcess::readyReadStandardOutput, this,
                &MainWindow::processOutput);

        connect(m_bridge, &QProcess::readyReadStandardError, this, [this]() {
            QByteArray err = m_bridge->readAllStandardError();
            if (!err.isEmpty() && !err.contains("libpng warning")) {
                qDebug() << "BRIDGE ERR:" << err;
            }
        });

        connect(m_bridge,
                QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
                this, &MainWindow::stopMission);

        // Push current model states into bridge
        m_bridge->write(m_srEnabled   ? "SR_ON\n"   : "SR_OFF\n");
        m_bridge->write(m_rifeEnabled ? "RIFE_ON\n" : "RIFE_OFF\n");

    } else {
        stopMission();
    }
}

// ── Stop Mission ──────────────────────────────────────────────────────────────
void MainWindow::stopMission() {
    if (m_bridge) {
        m_bridge->write("QUIT\n");
        m_bridge->kill();
        m_bridge->waitForFinished(1000);
        m_bridge->deleteLater();
        m_bridge = nullptr;
    }
    m_buffer.clear();

    m_startBtn->setText(m_isEnglish ? "▶  START LINK" : "▶  開始連線");
    m_startBtn->setObjectName("");
    m_startBtn->setStyle(m_startBtn->style());

    m_statusLabel->setText(m_isEnglish ? "STANDING BY..." : "待命中...");
    m_statusLabel->setStyleSheet(
        QString("color: %1; font-weight: 700; font-size: 13px;"
                "background: %2; padding: 10px 18px; border-radius: 8px;"
                "border: 1px solid %3;")
            .arg(Styles::TEXT_MAIN, Styles::CARD_BG, Styles::BORDER));

    m_videoLabel->clear();
    m_fpsVal->setText("-");
    m_rssiVal->setText("-");
    m_successVal->setText("-");
    m_rateVal->setText("-");
    m_queueVal->setText("-");
}

// ── Toggle RealESRGAN ─────────────────────────────────────────────────────────
void MainWindow::toggleSR() {
    m_srEnabled = !m_srEnabled;
    if (m_srEnabled) {
        m_srBtn->setText(m_isEnglish ? "◆  REALESRGAN  [ ON ]" : "◆  REALESRGAN  [ 啟用 ]");
        m_srBtn->setObjectName("ToggleSR");
        if (m_bridge) m_bridge->write("SR_ON\n");
    } else {
        m_srBtn->setText(m_isEnglish ? "◆  REALESRGAN  [ OFF ]" : "◆  REALESRGAN  [ 停用 ]");
        m_srBtn->setObjectName("ToggleSROff");
        if (m_bridge) m_bridge->write("SR_OFF\n");
    }
    m_srBtn->setStyle(m_srBtn->style());
}

// ── Toggle RIFE ───────────────────────────────────────────────────────────────
void MainWindow::toggleRIFE() {
    m_rifeEnabled = !m_rifeEnabled;
    if (m_rifeEnabled) {
        m_rifeBtn->setText(m_isEnglish ? "◈  RIFE INTERPOLATION  [ ON ]" : "◈  RIFE 補幀  [ 啟用 ]");
        m_rifeBtn->setObjectName("ToggleRIFE");
        if (m_bridge) m_bridge->write("RIFE_ON\n");
    } else {
        m_rifeBtn->setText(m_isEnglish ? "◈  RIFE INTERPOLATION  [ OFF ]" : "◈  RIFE 補幀  [ 停用 ]");
        m_rifeBtn->setObjectName("ToggleRIFEOff");
        if (m_bridge) m_bridge->write("RIFE_OFF\n");
    }
    m_rifeBtn->setStyle(m_rifeBtn->style());
}

// ============================================================================
//  SYNC HEADER  (must match SYNC_FORMAT in rx_bridge_v4.py)
//  '<4sIIfiiiiii'
//   sync[4], width, height, fps, rssi, sr_enabled, right, wrong,
//   rife_enabled, rife_ok
// ============================================================================
#pragma pack(push, 1)
struct SyncHeader {
    char     sync[4];
    uint32_t w;
    uint32_t h;
    float    fps;
    int32_t  rssi;
    int32_t  sr;
    int32_t  right;
    int32_t  wrong;
    int32_t  rifeEnabled;  // ← NEW
    int32_t  rifeOk;       // ← NEW
};
#pragma pack(pop)

// ── Process stdout from bridge ────────────────────────────────────────────────
void MainWindow::processOutput() {
    m_buffer.append(m_bridge->readAllStandardOutput());

    bool hasNewFrame = false;
    QImage latestImage;
    
    // Telemetry state to update after the loop
    float t_fps = 0;
    int t_rssi = 0, t_right = 0, t_wrong = 0;
    bool t_rifeOk = false, t_rifeEnabled = false, t_sr = false;

    while (true) {
        if (m_buffer.size() < (int)sizeof(SyncHeader))
            break;

        int syncIdx = m_buffer.indexOf("SYNC");
        if (syncIdx == -1) {
            m_buffer = m_buffer.right(3);
            break;
        }
        if (syncIdx > 0) {
            m_buffer.remove(0, syncIdx);
            if (m_buffer.size() < (int)sizeof(SyncHeader))
                break;
        }

        auto *hdr = reinterpret_cast<SyncHeader *>(m_buffer.data());
        int w = hdr->w;
        int h = hdr->h;
        int expectedSize = 0;
        bool isJpeg = false;

        if (w == 0 && h > 0) {
            // JPEG Mode: w = 0, h = payload length
            isJpeg = true;
            expectedSize = sizeof(SyncHeader) + h;
        } else {
            // Raw RGB Mode (legacy)
            expectedSize = sizeof(SyncHeader) + w * h * 3;
        }

        if (expectedSize <= (int)sizeof(SyncHeader) || expectedSize > 20'000'000) {
            m_buffer.remove(0, 4); // corrupted header
            continue;
        }
        if (m_buffer.size() < expectedSize)
            break;

        // Save latest telemetry
        t_fps = hdr->fps;
        t_rssi = hdr->rssi;
        t_right = hdr->right;
        t_wrong = hdr->wrong;
        t_rifeOk = hdr->rifeOk;
        t_rifeEnabled = hdr->rifeEnabled;
        t_sr = hdr->sr;

        // Extract frame
        const uchar *imgData = reinterpret_cast<const uchar *>(m_buffer.data() + sizeof(SyncHeader));
        if (isJpeg) {
            latestImage.loadFromData(imgData, h, "JPG");
        } else {
            latestImage = QImage(imgData, w, h, QImage::Format_RGB888).copy();
        }
        
        hasNewFrame = true;
        m_buffer.remove(0, expectedSize);
    }

    // ── Update UI only once per chunk to prevent Event Loop stutter ──────────
    if (hasNewFrame) {
        m_fpsVal->setText(QString::number(t_fps, 'f', 1));
        m_rssiVal->setText(QString::number(t_rssi));
        m_successVal->setText(QString::number(t_right));

        int total = t_right + t_wrong;
        m_rateVal->setText(total > 0 ? QString::number((t_right * 100.0f) / total, 'f', 1) + "%" : "N/A");

        QString queueStr = "";
        queueStr += t_rifeOk ? (t_rifeEnabled ? "RIFE ✓ " : "RIFE ✗ ") : "RIFE N/A ";
        queueStr += t_sr ? "SR ✓" : "SR ✗";
        m_queueVal->setText(queueStr);
        m_queueVal->setStyleSheet(
            QString("color: %1; font-size: 12px; font-weight: 600; font-family: 'Consolas';").arg(Styles::CYAN));

        if (!latestImage.isNull()) {
            QSize sz = m_videoLabel->parentWidget()->size();
            m_videoLabel->setPixmap(
                QPixmap::fromImage(latestImage)
                    .scaled(sz.width() - 10, sz.height() - 10,
                            Qt::KeepAspectRatio, Qt::FastTransformation));
        }
    }
}
