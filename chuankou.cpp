#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <unistd.h>

#define BUFFER_SIZE 256
// 初始化UART4
int init_uart4(const char *device)
{
    int uart4_fd = open(device, O_RDWR | O_NOCTTY | O_NDELAY);
    if (uart4_fd == -1)
    {
        perror("Unable to open UART");
        return -1;
    }

    struct termios options;
    tcgetattr(uart4_fd, &options);

    // 设置波特率
    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);

    // 设置8位数据位，1位停止位，无奇偶校验
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;

    // 使能接收
    options.c_cflag |= (CLOCAL | CREAD);

    // 设置为原始模式
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_oflag &= ~OPOST;

    tcsetattr(uart4_fd, TCSANOW, &options);

    return uart4_fd;
}

void send_hex_message(int uart_fd, const char *message)
{
    // 将字符串消息转换为16进制字节数组
    unsigned char hex_message[100];
    int len = strlen(message) / 2; // 16进制字符串的长度除以2
    for (int i = 0; i < len; ++i)
    {
        sscanf(&message[i * 2], "%2hhx", &hex_message[i]);
    }
    // 通过串口发送字节数组
    write(uart_fd, hex_message, len);
}
// 发送信息
int send_message(int uart4_fd, const char *message)
{
    int count = write(uart4_fd, message, strlen(message));
    if (count < 0)
    {
        perror("UART TX error");
        return -1;
    }
    return count;
}

void convert_to_hex_string(const char *buffer, int length, char *hex_string) {
    for (int i = 0; i < length; i++) {
        sprintf(hex_string + i * 2, "%02X", (unsigned char)buffer[i]);
    }
}

// 接收并处理串口数据的线程函数
void* receive_thread(void* arg) {
    int uart4_fd = *(int*)arg;
    char buffer[BUFFER_SIZE];
    int bytes_read;
    const char* target_string = "EF01000000000100030203A5";
    char hex_string[BUFFER_SIZE * 2];

    while (!quit) {
        memset(buffer, 0, BUFFER_SIZE);
        memset(hex_string, 0, sizeof(hex_string));

        bytes_read = read(uart4_fd, buffer, BUFFER_SIZE - 1);

        if (bytes_read > 0) {
            convert_to_hex_string(buffer, bytes_read, hex_string);

            // 检查接收到的消息是否包含目标字符串
            if (strstr(hex_string, target_string) != NULL) {
                send_hex_message(uart4_fd, "EF010000000001001B02226D6F64656C223A226F6B22A5");
            }
        } else if (bytes_read < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            } else {
                continue;
            }
        } else {
            continue;
        }
    }

    return NULL;
}
// int main() {
//     const char *uart_device = "/dev/ttyS4"; // UART4设备文件路径
//     int uart4_fd = init_uart4(uart_device);
//     if (uart4_fd == -1) {
//         return -1;
//     }

//     const char *message = "EF 01 00 00 00 00 01 00 1A   01 7B 22 64 6D 6F 64 2D 72 65 61 72 22   3A 22 64 61 6E 67 65 72 22 7D 00 B5 66 ";
//     int result = send_message(uart4_fd, message);
//     if (result > 0) {
//         printf("Message sent: %s\n", message);
//     }

//     close(uart4_fd);
//     return 0;
// }
