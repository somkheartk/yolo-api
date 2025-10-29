# คู่มือการ Deploy YOLO API บน Digital Ocean

คู่มือนี้จะแนะนำขั้นตอนการ deploy YOLO Object Detection API บน Digital Ocean อย่างละเอียด

## สารบัญ

1. [ภาพรวม](#ภาพรวม)
2. [ข้อกำหนดเบื้องต้น](#ข้อกำหนดเบื้องต้น)
3. [การสร้าง Droplet](#การสร้าง-droplet)
4. [การตั้งค่า Server เบื้องต้น](#การตั้งค่า-server-เบื้องต้น)
5. [การติดตั้ง Docker](#การติดตั้ง-docker)
6. [การ Deploy Application](#การ-deploy-application)
7. [การตั้งค่า Domain และ SSL](#การตั้งค่า-domain-และ-ssl)
8. [การตั้งค่า Nginx Reverse Proxy](#การตั้งค่า-nginx-reverse-proxy)
9. [การตั้งค่า Firewall](#การตั้งค่า-firewall)
10. [Monitoring และ Logging](#monitoring-และ-logging)
11. [Backup และ Recovery](#backup-และ-recovery)
12. [Performance Tuning](#performance-tuning)
13. [Troubleshooting](#troubleshooting)

---

## ภาพรวม

Digital Ocean เป็น cloud provider ที่เหมาะสำหรับการ deploy API เพราะ:
- 💰 ราคาไม่แพง ($4-6/เดือนสำหรับเริ่มต้น)
- 🚀 ตั้งค่าง่าย straightforward
- 📊 มี monitoring tools
- 🔄 Scale ได้ง่าย
- 📍 มี data centers ในเอเชีย

### สถาปัตยกรรม

```
Internet
    ↓
Domain (yourdomain.com)
    ↓
SSL Certificate (Let's Encrypt)
    ↓
Nginx (Reverse Proxy)
    ↓
Docker Container (YOLO API)
    ↓
YOLOv8 Model
```

---

## ข้อกำหนดเบื้องต้น

### 1. บัญชี Digital Ocean
- สมัครที่ https://www.digitalocean.com
- เติมเงินหรือเพิ่ม payment method

### 2. Domain Name (Optional แต่แนะนำ)
- ซื้อ domain จาก registrar (Namecheap, GoDaddy, etc.)
- หรือใช้ IP address โดยตรง

### 3. โมเดลที่พร้อมใช้งาน
- มีไฟล์ .pt model ที่ train เสร็จแล้ว
- หรือจะใช้ pretrained model ก็ได้

---

## การสร้าง Droplet

### ขั้นตอนที่ 1: สร้าง Droplet

1. **Login เข้า Digital Ocean Console**
   - ไปที่ https://cloud.digitalocean.com

2. **Create > Droplets**

3. **เลือก Image:**
   - **Recommended**: Ubuntu 22.04 (LTS) x64

4. **เลือกขนาด Droplet:**

**สำหรับ Development/Testing:**
```
Basic Plan
CPU: 1 vCPU
RAM: 2 GB
Storage: 50 GB SSD
Transfer: 2 TB
ราคา: $12/month (อาจมีการเปลี่ยนแปลง - ตรวจสอบราคาปัจจุบันที่ digitalocean.com)
```

**สำหรับ Production (แนะนำ):**
```
Basic Plan
CPU: 2 vCPUs
RAM: 4 GB
Storage: 80 GB SSD
Transfer: 4 TB
ราคา: $24/month (อาจมีการเปลี่ยนแปลง - ตรวจสอบราคาปัจจุบันที่ digitalocean.com)
```

**สำหรับ High Traffic:**
```
CPU-Optimized
CPU: 4 vCPUs
RAM: 8 GB
Storage: 100 GB SSD
ราคา: $48/month (อาจมีการเปลี่ยนแปลง - ตรวจสอบราคาปัจจุบันที่ digitalocean.com)
```

5. **เลือก Data Center Region:**
   - Singapore (SGP1) - ใกล้ไทยที่สุด
   - หรือ Frankfurt (FRA1) - ทางเลือกรอง

6. **Authentication:**
   - **SSH Key** (แนะนำ): อัพโหลด public SSH key
   - หรือ **Password**: ตั้ง root password

7. **Additional Options:**
   - ✅ Enable **Monitoring** (free)
   - ✅ Enable **IPv6**
   - ❌ ไม่ต้องเลือก **Backups** ในตอนนี้ (เสียเงินเพิ่ม 20%)

8. **Hostname:**
   - ตั้งชื่อ เช่น `yolo-api-prod`

9. **Create Droplet**

รอ 1-2 นาที droplet จะพร้อมใช้งาน คุณจะได้รับ:
- IP Address (เช่น 159.89.xxx.xxx)
- Root password (ถ้าไม่ได้ใช้ SSH key)

---

## การตั้งค่า Server เบื้องต้น

### ขั้นตอนที่ 1: เชื่อมต่อกับ Server

**ถ้าใช้ SSH Key:**
```bash
ssh root@YOUR_DROPLET_IP
```

**ถ้าใช้ Password:**
```bash
ssh root@YOUR_DROPLET_IP
# ใส่ password ที่ได้รับทาง email
```

### ขั้นตอนที่ 2: Update System

```bash
# Update package lists
apt update

# Upgrade all packages
apt upgrade -y

# Install essential tools
apt install -y curl wget git vim ufw
```

### ขั้นตอนที่ 3: สร้าง User ใหม่ (Security Best Practice)

```bash
# สร้าง user ใหม่
adduser yoloapi

# เพิ่ม user เข้า sudo group
usermod -aG sudo yoloapi

# เพิ่ม user เข้า docker group (จะติดตั้ง docker ในขั้นต่อไป)
usermod -aG docker yoloapi
```

### ขั้นตอนที่ 4: ตั้งค่า SSH สำหรับ User ใหม่

```bash
# Copy SSH keys จาก root ไปยัง user ใหม่
rsync --archive --chown=yoloapi:yoloapi ~/.ssh /home/yoloapi
```

### ขั้นตอนที่ 5: Configure SSH (เพิ่มความปลอดภัย)

```bash
# Edit SSH config
vim /etc/ssh/sshd_config
```

แก้ไขค่าต่อไปนี้:
```
# ปิดการ login ด้วย root
PermitRootLogin no

# ใช้ SSH key เท่านั้น (ถ้าคุณมี SSH key)
PasswordAuthentication no

# เปลี่ยน default port (optional แต่แนะนำ)
Port 2222

# Allow specific user
AllowUsers yoloapi
```

บันทึกและ restart SSH:
```bash
systemctl restart ssh
```

**หมายเหตุ:** หลังจากนี้ต้อง login ด้วย:
```bash
ssh -p 2222 yoloapi@YOUR_DROPLET_IP
```

### ขั้นตอนที่ 6: ตั้งค่า Swap (สำหรับ server ที่มี RAM น้อย)

```bash
# สร้าง swap file 2GB
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile

# ทำให้ swap persistent
echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab

# ตรวจสอบ
free -h
```

---

## การติดตั้ง Docker

### ขั้นตอนที่ 1: ติดตั้ง Docker

```bash
# เพิ่ม Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# เพิ่ม Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# ติดตั้ง Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# ตรวจสอบการติดตั้ง
docker --version
docker compose version
```

### ขั้นตอนที่ 2: Configure Docker

```bash
# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# ทดสอบ Docker
sudo docker run hello-world
```

### ขั้นตอนที่ 3: เพิ่ม user เข้า docker group

```bash
sudo usermod -aG docker $USER

# Logout และ login ใหม่เพื่อให้มีผล
exit
# ssh กลับเข้าไปใหม่
```

หลังจาก login กลับเข้ามา ทดสอบโดยไม่ต้องใช้ sudo:
```bash
docker ps
```

---

## การ Deploy Application

### ขั้นตอนที่ 1: Clone Repository

```bash
# สร้างโฟลเดอร์สำหรับ applications
mkdir -p ~/apps
cd ~/apps

# Clone repository
git clone https://github.com/somkheartk/yolo-api.git
cd yolo-api
```

### ขั้นตอนที่ 2: เตรียมโมเดล

**วิธีที่ 1: ใช้ Pretrained Model (ง่ายที่สุด)**
```bash
# API จะ download YOLOv8n โดยอัตโนมัติ
# ไม่ต้องทำอะไร
```

**วิธีที่ 2: Upload Custom Model**

จากเครื่อง local:
```bash
# Upload model จากเครื่อง local ไปยัง server
scp -P 2222 /path/to/your/model.pt yoloapi@YOUR_DROPLET_IP:~/apps/yolo-api/models/
```

หรือดาวน์โหลดจาก URL:
```bash
# บน server
cd ~/apps/yolo-api/models
wget https://your-url.com/model.pt -O custom_model.pt
```

### ขั้นตอนที่ 3: แก้ไข Configuration

แก้ไข `docker-compose.yml`:
```bash
vim docker-compose.yml
```

```yaml
services:
  yolo-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      # แก้ไข path ของโมเดล
      - MODEL_PATH=/app/models/yolov8n.pt
      # หรือ custom model
      # - MODEL_PATH=/app/models/custom_model.pt
    restart: unless-stopped
    # เพิ่ม resource limits
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

### ขั้นตอนที่ 4: Build และ Run

```bash
# Build Docker image
docker compose build

# Run container
docker compose up -d

# ตรวจสอบสถานะ
docker compose ps

# ดู logs
docker compose logs -f
```

### ขั้นตอนที่ 5: ทดสอบ API

```bash
# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model-info
```

ถ้าทำงานปกติ ควรได้ response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/app/models/yolov8n.pt"
}
```

---

## การตั้งค่า Domain และ SSL

### ขั้นตอนที่ 1: ตั้งค่า DNS

ไปที่ domain registrar ของคุณ (Namecheap, GoDaddy, etc.) และเพิ่ม A record:

```
Type: A
Host: @ (หรือ subdomain เช่น api)
Value: YOUR_DROPLET_IP
TTL: Auto หรือ 300
```

ตัวอย่าง:
- `yourdomain.com` → `159.89.xxx.xxx`
- `api.yourdomain.com` → `159.89.xxx.xxx`

รอ DNS propagate (5-30 นาที)

ทดสอบ:
```bash
nslookup yourdomain.com
# หรือ
dig yourdomain.com
```

### ขั้นตอนที่ 2: ติดตั้ง Nginx

```bash
sudo apt install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

ทดสอบ: เปิด browser ไปที่ `http://YOUR_DROPLET_IP` ควรเห็นหน้า Welcome to Nginx

### ขั้นตอนที่ 3: ติดตั้ง Certbot (Let's Encrypt)

```bash
# ติดตั้ง Certbot
sudo apt install -y certbot python3-certbot-nginx
```

### ขั้นตอนที่ 4: ขอ SSL Certificate

```bash
# ขอ certificate สำหรับ domain
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# หรือสำหรับ subdomain
sudo certbot --nginx -d api.yourdomain.com
```

ตอบคำถาม:
1. Email: ใส่ email ของคุณ
2. Terms of Service: `A` (Agree)
3. Share email: `N` (No)

Certbot จะ:
- ขอ certificate จาก Let's Encrypt
- แก้ไข Nginx config อัตโนมัติ
- Redirect HTTP → HTTPS

ทดสอบ: เปิด `https://yourdomain.com` ควรเห็น 🔒

### ขั้นตอนที่ 5: ตั้งค่า Auto-renewal

```bash
# ทดสอบ renewal
sudo certbot renew --dry-run

# Certbot จะตั้ง cron job อัตโนมัติ ตรวจสอบได้จาก:
sudo systemctl list-timers
```

---

## การตั้งค่า Nginx Reverse Proxy

### ขั้นตอนที่ 1: สร้าง Nginx Configuration

```bash
sudo vim /etc/nginx/sites-available/yolo-api
```

เพิ่ม configuration:

```nginx
# Upstream definition
upstream yolo_api {
    server localhost:8000;
}

# HTTP - Redirect to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name api.yourdomain.com;

    return 301 https://$server_name$request_uri;
}

# HTTPS
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name api.yourdomain.com;

    # SSL certificates (Certbot จะเพิ่มให้อัตโนมัติ)
    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Logging
    access_log /var/log/nginx/yolo-api-access.log;
    error_log /var/log/nginx/yolo-api-error.log;

    # Client body size (สำหรับ upload รูปภาพ)
    client_max_body_size 10M;

    # Timeouts
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;

    # Proxy settings
    location / {
        proxy_pass http://yolo_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (ถ้าต้องการ)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Health check endpoint
    location /health {
        proxy_pass http://yolo_api/health;
        access_log off;
    }
}
```

### ขั้นตอนที่ 2: Enable Site

```bash
# สร้าง symbolic link
sudo ln -s /etc/nginx/sites-available/yolo-api /etc/nginx/sites-enabled/

# ลบ default site (optional)
sudo rm /etc/nginx/sites-enabled/default

# ทดสอบ configuration
sudo nginx -t

# Reload Nginx
sudo systemctl reload nginx
```

### ขั้นตอนที่ 3: ทดสอบ

```bash
# จาก server
curl https://api.yourdomain.com/health

# จากเครื่อง local
curl https://api.yourdomain.com/health
```

---

## การตั้งค่า Firewall

### ขั้นตอนที่ 1: Configure UFW (Uncomplicated Firewall)

```bash
# Reset firewall (ถ้าต้องการ)
sudo ufw --force reset

# Default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (ใช้ port ที่เปลี่ยนไว้)
sudo ufw allow 2222/tcp comment 'SSH'

# Allow HTTP และ HTTPS
sudo ufw allow 80/tcp comment 'HTTP'
sudo ufw allow 443/tcp comment 'HTTPS'

# Enable firewall
sudo ufw enable

# ตรวจสอบสถานะ
sudo ufw status verbose
```

### ขั้นตอนที่ 2: Digital Ocean Firewall (เพิ่มเติม)

ใน Digital Ocean Console:

1. ไปที่ **Networking** → **Firewalls**
2. **Create Firewall**
3. ตั้งค่า Inbound Rules:
   ```
   Type        Protocol    Port Range    Sources
   SSH         TCP         2222          Your IP
   HTTP        TCP         80            All IPv4, All IPv6
   HTTPS       TCP         443           All IPv4, All IPv6
   ```
4. เลือก Droplet ที่จะใช้
5. **Create Firewall**

---

## Monitoring และ Logging

### ขั้นตอนที่ 1: ตั้งค่า System Monitoring

**Install monitoring tools:**
```bash
sudo apt install -y htop iotop nethogs
```

**ตรวจสอบ resources:**
```bash
# CPU และ RAM
htop

# Disk usage
df -h

# Disk I/O
sudo iotop

# Network usage
sudo nethogs

# Docker stats
docker stats
```

### ขั้นตอนที่ 2: Application Logs

**Docker logs:**
```bash
# ดู logs แบบ realtime
docker compose logs -f

# ดู logs ล่าสุด 100 บรรทัด
docker compose logs --tail=100

# ดู logs ของ specific container
docker logs yolo-api-yolo-api-1 -f
```

**Nginx logs:**
```bash
# Access logs
sudo tail -f /var/log/nginx/yolo-api-access.log

# Error logs
sudo tail -f /var/log/nginx/yolo-api-error.log
```

### ขั้นตอนที่ 3: Log Rotation

สร้าง log rotation config:
```bash
sudo vim /etc/logrotate.d/yolo-api
```

```
/var/log/nginx/yolo-api-*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        [ -f /var/run/nginx.pid ] && kill -USR1 `cat /var/run/nginx.pid`
    endscript
}
```

### ขั้นตอนที่ 4: Setup Monitoring Script

สร้าง monitoring script:
```bash
vim ~/monitor.sh
```

```bash
#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== YOLO API Health Monitor ==="
echo ""

# Check if API is running
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$API_STATUS" = "200" ]; then
    echo -e "${GREEN}✓${NC} API is running (HTTP $API_STATUS)"
else
    echo -e "${RED}✗${NC} API is down (HTTP $API_STATUS)"
fi

# Check Docker container
CONTAINER_STATUS=$(docker inspect -f '{{.State.Status}}' yolo-api-yolo-api-1 2>/dev/null)
if [ "$CONTAINER_STATUS" = "running" ]; then
    echo -e "${GREEN}✓${NC} Docker container is running"
else
    echo -e "${RED}✗${NC} Docker container is not running"
fi

# Check disk space
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 80 ]; then
    echo -e "${GREEN}✓${NC} Disk usage: ${DISK_USAGE}%"
else
    echo -e "${YELLOW}⚠${NC} Disk usage: ${DISK_USAGE}% (Warning: >80%)"
fi

# Check memory
MEM_USAGE=$(free | awk 'NR==2 {printf "%.0f", $3/$2*100}')
if [ "$MEM_USAGE" -lt 90 ]; then
    echo -e "${GREEN}✓${NC} Memory usage: ${MEM_USAGE}%"
else
    echo -e "${YELLOW}⚠${NC} Memory usage: ${MEM_USAGE}% (Warning: >90%)"
fi

echo ""
echo "Last 5 access logs:"
sudo tail -5 /var/log/nginx/yolo-api-access.log

echo ""
echo "==================================="
```

ทำให้ executable:
```bash
chmod +x ~/monitor.sh
```

ใช้งาน:
```bash
./monitor.sh
```

### ขั้นตอนที่ 5: Setup Cron Job สำหรับ Monitoring

```bash
crontab -e
```

เพิ่ม:
```
# Monitor API every 5 minutes
*/5 * * * * ~/monitor.sh >> ~/monitor.log 2>&1

# Restart if down (optional)
*/10 * * * * docker ps | grep yolo-api || docker compose -f ~/apps/yolo-api/docker-compose.yml up -d
```

---

## Backup และ Recovery

### ขั้นตอนที่ 1: Manual Backup

**Backup โมเดล:**
```bash
# Backup models
tar -czf ~/backups/models-$(date +%Y%m%d).tar.gz ~/apps/yolo-api/models/

# Download จาก server ไปยัง local
scp -P 2222 yoloapi@YOUR_DROPLET_IP:~/backups/models-*.tar.gz ./
```

**Backup configuration:**
```bash
# Backup configs
tar -czf ~/backups/configs-$(date +%Y%m%d).tar.gz \
    ~/apps/yolo-api/docker-compose.yml \
    /etc/nginx/sites-available/yolo-api
```

### ขั้นตอนที่ 2: Automated Backup Script

```bash
vim ~/backup.sh
```

```bash
#!/bin/bash

BACKUP_DIR=~/backups
DATE=$(date +%Y%m%d_%H%M%S)

# สร้าง backup directory
mkdir -p $BACKUP_DIR

# Backup models
echo "Backing up models..."
tar -czf $BACKUP_DIR/models-$DATE.tar.gz ~/apps/yolo-api/models/

# Backup configs
echo "Backing up configs..."
tar -czf $BACKUP_DIR/configs-$DATE.tar.gz \
    ~/apps/yolo-api/docker-compose.yml \
    /etc/nginx/sites-available/yolo-api

# ลบ backup เก่าที่เกิน 7 วัน
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
```

```bash
chmod +x ~/backup.sh
```

ตั้ง cron job:
```bash
crontab -e
```

เพิ่ม:
```
# Daily backup at 2 AM
0 2 * * * ~/backup.sh >> ~/backup.log 2>&1
```

### ขั้นตอนที่ 3: Digital Ocean Snapshots

ใน Digital Ocean Console:
1. ไปที่ Droplet → **Snapshots**
2. **Take Snapshot**
3. ตั้งชื่อ เช่น `yolo-api-before-update-2024-01-15`
4. ใช้เวลา 1-2 นาที

**Automated Snapshots:**
- ไปที่ Droplet Settings → **Backups**
- Enable Automated Backups (เพิ่ม 20% ของราคา droplet)

### ขั้นตอนที่ 4: Recovery Process

**Recovery จาก backup:**
```bash
# 1. Stop container
cd ~/apps/yolo-api
docker compose down

# 2. Restore models
tar -xzf ~/backups/models-YYYYMMDD.tar.gz -C ~/

# 3. Restore configs
tar -xzf ~/backups/configs-YYYYMMDD.tar.gz -C ~/

# 4. Restart container
docker compose up -d
```

**Recovery จาก snapshot:**
1. ใน Digital Ocean Console
2. สร้าง Droplet ใหม่จาก Snapshot
3. Update DNS record ให้ชี้ไปที่ IP ใหม่

---

## Performance Tuning

### 1. Nginx Tuning

แก้ไข `/etc/nginx/nginx.conf`:

```nginx
user www-data;
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    # Basic Settings
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    server_tokens off;

    # Buffer Settings
    client_body_buffer_size 128k;
    client_max_body_size 10m;
    client_header_buffer_size 1k;
    large_client_header_buffers 4 4k;
    output_buffers 1 32k;
    postpone_output 1460;

    # Gzip Settings
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript 
               application/json application/javascript application/xml+rss 
               application/rss+xml font/truetype font/opentype 
               application/vnd.ms-fontobject image/svg+xml;

    # Logging
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    # Include configs
    include /etc/nginx/mime.types;
    include /etc/nginx/conf.d/*.conf;
    include /etc/nginx/sites-enabled/*;
}
```

Reload:
```bash
sudo nginx -t && sudo nginx -s reload
```

### 2. Docker Tuning

แก้ไข `docker-compose.yml`:

```yaml
services:
  yolo-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/yolov8n.pt
    restart: unless-stopped
    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 3G
        reservations:
          cpus: '1'
          memory: 1G
    # Healthcheck
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 3. System Tuning

แก้ไข `/etc/sysctl.conf`:

```bash
sudo vim /etc/sysctl.conf
```

เพิ่ม:
```
# Network tuning
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.ip_local_port_range = 1024 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 15

# File descriptors
fs.file-max = 65535
```

Apply:
```bash
sudo sysctl -p
```

### 4. Model Optimization

**ใช้ ONNX สำหรับ production:**

```bash
# บน server
cd ~/apps/yolo-api

# สร้าง script สำหรับ export ONNX
cat > export_onnx.py << 'EOF'
from ultralytics import YOLO

model = YOLO('models/yolov8n.pt')
model.export(format='onnx', dynamic=True, simplify=True)
print("Model exported to ONNX format")
EOF

# รัน export
docker run --rm -v $(pwd):/app -w /app python:3.10-slim bash -c "
  pip install ultralytics && python export_onnx.py
"

# Update docker-compose.yml ให้ใช้ ONNX model
# MODEL_PATH=/app/models/yolov8n.onnx
```

---

## Troubleshooting

### ปัญหา: API ไม่ตอบ

**ตรวจสอบ:**
```bash
# 1. Container status
docker ps
docker compose logs

# 2. Port listening
sudo netstat -tlnp | grep 8000

# 3. Firewall
sudo ufw status
```

**แก้ไข:**
```bash
# Restart container
docker compose restart

# หรือ rebuild
docker compose down
docker compose up -d --build
```

### ปัญหา: Out of Memory

**ตรวจสอบ:**
```bash
free -h
docker stats
```

**แก้ไข:**
```bash
# 1. เพิ่ม swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 2. ลดขนาด batch หรือใช้โมเดลที่เล็กกว่า
# 3. Upgrade droplet
```

### ปัญหา: SSL Certificate Error

**ตรวจสอบ:**
```bash
sudo certbot certificates
```

**แก้ไข:**
```bash
# Renew certificate
sudo certbot renew

# หรือ renew specific domain
sudo certbot renew --cert-name yourdomain.com
```

### ปัญหา: Slow Response

**ตรวจสอบ:**
```bash
# 1. CPU usage
htop

# 2. Network
ping yourdomain.com

# 3. API response time
curl -w "\nTime: %{time_total}s\n" http://localhost:8000/health
```

**แก้ไข:**
1. ใช้โมเดลที่เล็กกว่า (YOLOv8n)
2. Export เป็น ONNX/TensorRT
3. Upgrade droplet
4. ใช้ CDN

### ปัญหา: Disk Full

**ตรวจสอบ:**
```bash
df -h
du -sh /var/log/*
du -sh ~/.cache/*
```

**แก้ไข:**
```bash
# ลบ Docker images/containers ที่ไม่ใช้
docker system prune -a

# ลบ logs เก่า
sudo find /var/log -type f -name "*.log" -mtime +30 -delete

# Rotate logs
sudo logrotate -f /etc/logrotate.conf
```

---

## สรุป

คุณได้เรียนรู้:
- ✅ การสร้างและตั้งค่า Digital Ocean Droplet
- ✅ การติดตั้ง Docker และ dependencies
- ✅ การ deploy YOLO API
- ✅ การตั้งค่า domain และ SSL
- ✅ การตั้งค่า Nginx reverse proxy
- ✅ การตั้งค่า firewall และความปลอดภัย
- ✅ Monitoring และ logging
- ✅ Backup และ recovery
- ✅ Performance tuning
- ✅ Troubleshooting

### Quick Reference Commands

```bash
# ดูสถานะ
docker compose ps
docker compose logs -f
./monitor.sh

# Restart API
docker compose restart

# Update code
cd ~/apps/yolo-api
git pull
docker compose up -d --build

# Backup
./backup.sh

# ดู logs
docker compose logs --tail=100
sudo tail -f /var/log/nginx/yolo-api-access.log

# ตรวจสอบ resources
htop
docker stats
df -h
```

### Next Steps

1. **Setup Monitoring**: ใช้ tools เช่น Grafana, Prometheus
2. **Load Balancing**: ถ้ามี traffic สูง
3. **CI/CD**: Setup automated deployment
4. **Rate Limiting**: จำกัดจำนวน requests
5. **API Authentication**: เพิ่ม API keys

---

## เอกสารเพิ่มเติม

- [Digital Ocean Documentation](https://docs.digitalocean.com)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Docker Documentation](https://docs.docker.com)
- [Let's Encrypt](https://letsencrypt.org/docs/)
- [YOLO API Development Guide](DEVELOPMENT.md)
