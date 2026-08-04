#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QComboBox>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QProcess>
#include <QPushButton>


class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

private slots:
  void toggleMission();
  void processOutput();
  void refreshPorts();
  void toggleSR();
  void toggleRIFE();
  void toggleLanguage();

private:
  void setupUi();
  void setupStyles();
  void stopMission();
  void updateUITexts();

  QProcess *m_bridge = nullptr;
  QByteArray m_buffer;

  QLabel *m_videoLabel;
  QComboBox *m_portCombo;
  QPushButton *m_startBtn;
  QPushButton *m_srBtn;
  QPushButton *m_rifeBtn;
  QLabel *m_statusLabel;

  // Telemetry labels
  QLabel *m_fpsVal;
  QLabel *m_rssiVal;
  QLabel *m_successVal;
  QLabel *m_rateVal;
  QLabel *m_queueVal;

  bool m_isEnglish = true;
  QLabel *m_logoMain;
  QLabel *m_logoSub;
  QLabel *m_feedLabel;
  QLabel *m_telLabel;
  QLabel *m_ctrlLabel;
  QLabel *m_aiLabel;
  QPushButton *m_langBtn;
  
  QLabel *m_yoloTitle;
  QLabel *m_yoloVal;

  class InfoCard *m_fpsCard;
  class InfoCard *m_rssiCard;
  class InfoCard *m_successCard;
  class InfoCard *m_rateCard;
  class InfoCard *m_queueCard;

  bool m_srEnabled   = true;
  bool m_rifeEnabled = true;
};
#endif // MAINWINDOW_H
