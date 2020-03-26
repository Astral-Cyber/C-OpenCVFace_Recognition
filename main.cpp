#include "MyTcpServer.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	MyTcpServer w;
	w.show();
	return a.exec();
}
