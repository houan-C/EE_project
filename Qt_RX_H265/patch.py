import os

filepath = 'd:/Qt_project/RX_H265/mainwindow.cpp'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. InfoCard
content = content.replace(
    '    QLabel *valLbl;\n    InfoCard(QString title, QString unit = "") : QFrame() {\n',
    '    QLabel *valLbl;\n    QLabel *titleLbl;\n    InfoCard(QString title, QString unit = "") : QFrame() {\n'
)
content = content.replace(
    '        auto titleLbl = new QLabel(title.toUpper());\n',
    '        titleLbl = new QLabel(title.toUpper());\n'
)

# 2. Header
content = content.replace(
    '    auto logoSub = new QLabel("RECEIVER");\n',
    '    m_logoSub = new QLabel("RECEIVER");\n'
)
content = content.replace(
    '    logoRow->addWidget(logoSub);\n',
    '    logoRow->addWidget(m_logoSub);\n'
)
content = content.replace(
    '    logoRow->addWidget(logoSub);\n    logoRow->addStretch();\n    logoLayout->addLayout(logoRow);\n',
    '''    logoRow->addWidget(m_logoSub);
    logoRow->addStretch();
    
    m_langBtn = new QPushButton("🌐 EN / 中");
    m_langBtn->setFixedHeight(36);
    m_langBtn->setStyleSheet(QString("background: transparent; border: 1px solid %1; color: %2;").arg(Styles::BORDER, Styles::TEXT_MAIN));
    connect(m_langBtn, &QPushButton::clicked, this, &MainWindow::toggleLanguage);
    logoRow->addWidget(m_langBtn);

    logoLayout->addLayout(logoRow);
'''
)

content = content.replace(
    '    auto tagline = new QLabel("H.265 · RIFE INTERPOLATION · REALESRGAN SUPER-RESOLUTION");\n',
    '    m_tagline = new QLabel("H.265 · RIFE INTERPOLATION · REALESRGAN SUPER-RESOLUTION");\n'
)
content = content.replace(
    '    tagline->setStyleSheet(\n',
    '    m_tagline->setStyleSheet(\n'
)
content = content.replace(
    '    logoLayout->addWidget(tagline);\n',
    '    logoLayout->addWidget(m_tagline);\n'
)

# 3. Feed Label
content = content.replace(
    '    auto feedLabel = new QLabel("▶  PRIMARY FEED");\n',
    '    m_feedLabel = new QLabel("▶  PRIMARY FEED");\n'
)
content = content.replace(
    '    feedLabel->setStyleSheet(\n',
    '    m_feedLabel->setStyleSheet(\n'
)
content = content.replace(
    '    leftSide->addWidget(feedLabel);\n',
    '    leftSide->addWidget(m_feedLabel);\n'
)

# 4. Telemetry Label and InfoCards
content = content.replace(
    '    auto telLabel = new QLabel("TELEMETRY");\n',
    '    m_telLabel = new QLabel("TELEMETRY");\n'
)
content = content.replace(
    '    telLabel->setStyleSheet(\n',
    '    m_telLabel->setStyleSheet(\n'
)
content = content.replace(
    '    rightSide->addWidget(telLabel);\n',
    '    rightSide->addWidget(m_telLabel);\n'
)

content = content.replace(
    '''    auto fpsCard     = new InfoCard("Decoded FPS",     "FPS");
    auto rssiCard    = new InfoCard("Link RSSI",       "dBm");
    auto successCard = new InfoCard("Frames OK",       "FRM");
    auto rateCard    = new InfoCard("Success Rate",    "%");
    auto queueCard   = new InfoCard("Display Queue",   "Q");

    m_fpsVal     = fpsCard->valLbl;
    m_rssiVal    = rssiCard->valLbl;
    m_successVal = successCard->valLbl;
    m_rateVal    = rateCard->valLbl;
    m_queueVal   = queueCard->valLbl;

    grid->addWidget(fpsCard,     0, 0);
    grid->addWidget(rssiCard,    0, 1);
    grid->addWidget(successCard, 1, 0);
    grid->addWidget(rateCard,    1, 1);
    grid->addWidget(queueCard,   2, 0, 1, 2);''',
    '''    m_fpsCard     = new InfoCard("Decoded FPS",     "FPS");
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
    grid->addWidget(m_queueCard,   2, 0, 1, 2);'''
)

# 5. Control Label and Remove refresh button
content = content.replace(
    '    auto ctrlLabel = new QLabel("CONTROL");\n',
    '    m_ctrlLabel = new QLabel("CONTROL");\n'
)
content = content.replace(
    '    ctrlLabel->setStyleSheet(\n',
    '    m_ctrlLabel->setStyleSheet(\n'
)
content = content.replace(
    '    rightSide->addWidget(ctrlLabel);\n',
    '    rightSide->addWidget(m_ctrlLabel);\n'
)

content = content.replace(
    '''    auto refreshBtn = new QPushButton("↺");
    refreshBtn->setFixedSize(36, 36);
    refreshBtn->setToolTip("Refresh COM ports");
    connect(refreshBtn, &QPushButton::clicked, this, &MainWindow::refreshPorts);
    cfgRow->addWidget(portIconLabel);
    cfgRow->addWidget(m_portCombo, 1);
    cfgRow->addWidget(refreshBtn);''',
    '''    cfgRow->addWidget(portIconLabel);
    cfgRow->addWidget(m_portCombo, 1);'''
)

# 6. AI Models label
content = content.replace(
    '    auto aiLabel = new QLabel("AI MODELS");\n',
    '    m_aiLabel = new QLabel("AI MODELS");\n'
)
content = content.replace(
    '    aiLabel->setStyleSheet(\n',
    '    m_aiLabel->setStyleSheet(\n'
)
content = content.replace(
    '    ctrlLayout->addWidget(aiLabel);\n',
    '    ctrlLayout->addWidget(m_aiLabel);\n'
)


# 7. Add updateUITexts and toggleLanguage, and modify toggle methods
new_methods = """
void MainWindow::toggleLanguage() {
    m_isEnglish = !m_isEnglish;
    updateUITexts();
}

void MainWindow::updateUITexts() {
    if (m_isEnglish) {
        m_logoSub->setText("RECEIVER");
        m_tagline->setText("H.265 · RIFE INTERPOLATION · REALESRGAN SUPER-RESOLUTION");
        if (m_statusLabel->text().contains("待命中")) m_statusLabel->setText("STANDING BY...");
        else if (m_statusLabel->text().contains("連線中")) m_statusLabel->setText("⬤  LINK ACTIVE: " + m_portCombo->currentText());
        else if (m_statusLabel->text().contains("錯誤")) m_statusLabel->setText("✖  ERROR: PYTHON PATH FAILED");
        
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
    } else {
        m_logoSub->setText("接收器");
        m_tagline->setText("H.265 · RIFE 補幀 · REALESRGAN 超解析度");
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
    }
}
"""

content = content.replace(
    'void MainWindow::refreshPorts() {',
    new_methods + '\nvoid MainWindow::refreshPorts() {'
)

# Apply state updates
content = content.replace(
    'm_statusLabel->setText("⬤  LINK ACTIVE: " + portName);',
    'm_statusLabel->setText(m_isEnglish ? ("⬤  LINK ACTIVE: " + portName) : ("⬤  連線中: " + portName));'
)
content = content.replace(
    'm_startBtn->setText("■  TERMINATE LINK");',
    'm_startBtn->setText(m_isEnglish ? "■  TERMINATE LINK" : "■  中斷連線");'
)
content = content.replace(
    'm_statusLabel->setText("✖  ERROR: PYTHON PATH FAILED");',
    'm_statusLabel->setText(m_isEnglish ? "✖  ERROR: PYTHON PATH FAILED" : "✖  錯誤: PYTHON 執行檔失敗");'
)
content = content.replace(
    'm_startBtn->setText("▶  START LINK");',
    'm_startBtn->setText(m_isEnglish ? "▶  START LINK" : "▶  開始連線");'
)
content = content.replace(
    'm_statusLabel->setText("STANDING BY...");',
    'm_statusLabel->setText(m_isEnglish ? "STANDING BY..." : "待命中...");'
)
content = content.replace(
    'm_srBtn->setText("◆  REALESRGAN  [ ON ]");',
    'm_srBtn->setText(m_isEnglish ? "◆  REALESRGAN  [ ON ]" : "◆  REALESRGAN  [ 啟用 ]");'
)
content = content.replace(
    'm_srBtn->setText("◆  REALESRGAN  [ OFF ]");',
    'm_srBtn->setText(m_isEnglish ? "◆  REALESRGAN  [ OFF ]" : "◆  REALESRGAN  [ 停用 ]");'
)
content = content.replace(
    'm_rifeBtn->setText("◈  RIFE INTERPOLATION  [ ON ]");',
    'm_rifeBtn->setText(m_isEnglish ? "◈  RIFE INTERPOLATION  [ ON ]" : "◈  RIFE 補幀  [ 啟用 ]");'
)
content = content.replace(
    'm_rifeBtn->setText("◈  RIFE INTERPOLATION  [ OFF ]");',
    'm_rifeBtn->setText(m_isEnglish ? "◈  RIFE INTERPOLATION  [ OFF ]" : "◈  RIFE 補幀  [ 停用 ]");'
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied.")
