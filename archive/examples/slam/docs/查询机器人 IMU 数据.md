**查询机器人 IMU 数据**

**请求**

编号: 1014 (0x03F6)

名称: robot_status_imu_req

描述: 查询机器人 IMU 数据

JSON 数据区: 无

**请求示例**

Plaintext  
5A 01 00 01 00 00 00 00 03 F6 00 00 00 00 00 00

**响应**

编号: 11014 (0x2B06)

名称: robot_status_imu_res

描述: 查询机器人 IMU 数据的响应

JSON 数据区: 见下表

|     |     |     |     |
| --- | --- | --- | --- |
| 字段名 | 类型  | 描述  | 可缺省 |
| imu_header | object | 这一帧imu数据的时间戳 | 是   |
| yaw | number | 偏航角，单位：rad | 是   |
| roll | number | 滚转角，单位：rad | 是   |
| pitch | number | 俯仰角，单位：rad | 是   |
| acc_x | number | 加速度计x轴ADC值 | 是   |
| acc_y | number | 加速度计y轴ADC值 | 是   |
| acc_z | number | 加速度计z轴ADC值 | 是   |
| rot_x | number | 陀螺仪x轴ADC值 | 是   |
| rot_y | number | 陀螺仪y轴ADC值 | 是   |
| rot_z | number | 陀螺仪z轴ADC值 | 是   |
| rot_off_x | int32 | 陀螺仪x轴偏置ADC值 | 是   |
| rot_off_y | int32 | 陀螺仪y轴偏置ADC值 | 是   |
| rot_off_z | int32 | 陀螺仪z轴偏置ADC值 | 是   |
| qx  | number | 四元数qx | 是   |
| qy  | number | 四元数qy | 是   |
| qz  | number | 四元数qz | 是   |
| qw  | number | 四元数qw | 是   |
| ret_code | number | API 错误码 | 是   |
| create_on | string | API 上传时间戳 | 是   |
| err_msg | string | 错误信息 | 是   |

**响应示例**

JSON  
{  
"acc_x":0,  
"acc_y":0,  
"acc_z":0,  
"create_on":"2023-03-24T14:58:46.293+0800",  
"imu_header":{  
"data_nsec":"16704707855595", // 小车开机到这帧数据的时间戳，单位为纳秒  
"frame_id":"/imu",  
"pub_nsec":"16704707855637", // 小车开机到这帧数据的时间戳，单位为纳秒  
"seq":"0"  
},  
"pitch":0,  
"qw":0,  
"qx":0,  
"qy":0,  
"qz":0,  
"ret_code":0,  
"roll":0,  
"rot_off_x":0,  
"rot_off_y":0,  
"rot_off_z":0,  
"rot_x":0,  
"rot_y":0,  
"rot_z":0,  
"yaw":-3.128697633743291  
}