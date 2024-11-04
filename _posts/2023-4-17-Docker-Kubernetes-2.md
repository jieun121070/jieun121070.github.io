---
title: "Docker/Kubernetes의 이해 - 2"
date: 2023-4-17
author: jieun
math: True
categories: [Model-Serving]
tags: [Docker, Kubernetes]
typora-root-url: ..
---

## 1. What is Kubernetes?

- 구글에서 개발하고 있는 컨테이너 배포, 확장, 운영 툴
- 사실상 현 시점 컨테이너 오케스트레이션 표준
- 대규모에 적합
- 장점
  - 다양한 생태계 구축되어 있음
  - 다양한 환경 지원
  - 훌륭한 확장성
- 단점
  - 알아야 할 내용이 너무 많아서 복잡한 클러스터를 구성하는게 아니라면 비추천. 득실을 잘 따져봐야 함
  - 설치가 너무 어려움. 가급적이면 클라우드 이용하는 것이 좋음
  - 환경이 무거움. 기본적으로 실행되는 프로그램이 많고, 그에 따라 기본 리소스 사용량이 많음
  - Docker Swarm에 비해 설정 파일이 복잡함

## 2. Kubernetes 특징/Architecture

### 2.1. 기본 기능

- 상태 관리: 상태를 선언하고 선언한 상태를 유지, 노드가 죽거나 컨테이너 응답이 없을 경우 자동 복구
- 스케줄링: 클러스터의 여러 노드 중 조건에 맞는 노드를 찾아 컨테이너를 배치
- 클러스터링: 가상 네트워크를 통해 하나의 서버에 있는 것처럼 통신
- 서비스 디스커버리: 서로 다른 서비스를 쉽게 찾고 통신할 수 있음
- 리소스 모니터링: cAdvisor를 통한 리소스 모니터링
- 스케일링: 리소스에 따라 자동으로 서비스를 조정함
- RollOut/RollBack: 배포/롤백 및 버전 관리

### 2.2. Architecture

![](/assets/img/docker/k8s_arch.png)

- **Master** - 명령을 받아 쿠버네티스 클러스터를 관리
  - `API Server` 운영자 및 내부 노드와 통신하기 위한 인터페이스. HTTP(S) RestAPI로 노출되어 있고, 모든 명령이 통하는 곳
  - `Controller Manager` 다양한 컨트롤러를 관리하고 API Server와 통신하여 작업을 수행
  - `etcd` 가볍고 빠른 분산형 key-value 저장소. 설정 및 상태를 저장
  - `Scheduler` 서비스를 리소스 상황에 맞게 적절한 노드에 배치하는 역할
- **Node** - 서비스(컨테이너)가 실행되는 서버. 마스터의 API Server와 통신하며 서비스 생성 및 상태 관리
  - `kubelet` 서비스(컨테이너)를 실행 및 중지하고 상태를 체크하여 계속해서 살아있는 상태로 관리. master와 worker node 간의 통신 역할
  - `Proxy` 네트워크 프록시와 load balancer 역할
  - `cAdvisor` 리소스 모니터링
  - `Docker` 도커뿐만 아니라 다양한 컨테이너 엔진을 지원

## 3. Kubernetes Object

- 모든 object는 특정 **namespace** 안에 존재
  - 하나의 물리적 클러스터를 논리적인 가상 클러스터로 관리
  - 지정하지 않으면 기본값은 default
- 모든 object는 유일한 **name**을 가짐
- **label**은 key-value로 구성하고, 여러 object를 의미있게 관리하기 위해 사용
- **annotation**은 주로 API의 설정값으로 활용

### 3.1. Pod

- 쿠버네티스는 컨테이너를 pod이라는 최소 개념으로 관리
- 하나의 pod은 여러 개의 컨테이너를 가질 수 있지만 대부분 한 개 또는 두 개.
  - 하나의 pod 안에 여러 개의 컨테이너가 있으면 `exec -it pod/example-redis -c app sh`와 같이 컨테이너를 지정해 주어야 함
- pod 내부에서 컨테이너는 네크워크(서로 localhost로 통신 가능)와 volume을 공유
- pod은 고유한 자체 IP가 있지만 직접 IP로 통신하지 않음. pod은 언제든지 죽을 수 있고, 다른 노드로 이동할 수 있고 확장되거나 축소될 수 있기 때문
  - pod에 연결할 수 있는 서비스를 만들고 해당 서비스의 IP, port를 통해 통신

### 3.2. Controller

- `ReplicaSet`(Replication Controller) pod을 확장할 때 사용
  - 동일한 pod의 개수가 많아질수록 pod을 일일이 정의하는 것은 매우 비효율적
  - pod이 삭제되거나 pod이 위치한 노드에 장애가 발생해 더이상 pod에 접근하지 못하게 되었을 때, 관리자가 직접 삭제 후 다시 생성하지 않는 한 해당 pod은 다시 복구되지 않음
  - ReplicaSet은 **정해진 수의 동일한 pod이 항상 실행되도록** 관리함
    - `replicas` 몇 개의 pod을 유지할 것인지 설정
    - `template` 어떤 pod을 생성할 것인지 설정.
    - 예를 들어 replicas는 2이고 template에서 label을 {service: example, type: app}로 설정한 상황에서 어떤 pod의 type key를 삭제하면 {service: example, type: app} label을 갖는 pod 1개가 자동 생성됨. 삭제했던 type key를 다시 추가하면 하나의 pod을 랜덤하게 골라 종료시킴
  - 노드 장애 등의 이유로 pod을 사용할 수 없다면 다른 노드에서 pod을 다시 생성
- `Deployment` **배포와 관련된 다양한 설정을 가지고 있는 컨트롤러**
  - 어떤 pod이 완전히 delete 되는 과정 `Deployment` > `ReplicaSet` > `Pod`
- `StatefulSet` pod을 만들 때 0부터 순서를 붙여줌. pod을 순서대로 생성하거나 순번에 따라 관리가 필요할 때 사용
- `DaemonSet` 모든 노드에 pod을 생성할 때 사용. 보통 모니터링을 위한 pod을 생성할 때 사용
- `Job` 한 번 실행되고 종료되는 서비스를 위한 컨트롤러. 보통 컨테이너의 상태가 종료되면 다시 살리려고 하지만 Job은 정상적으로 종료되었는지 체크함
- `Cronjob` 특정 주기로 실행되고 종료되는 Job을 위한 컨트롤러
- **pod은 컨테이너를 관리하고, controller는 pod을 관리하는 것**

### 3.3. Service

- `ClusterIP`
  - 여러 개의 pod을 중앙에서 load balancing하는 서비스
  - ClusterIP는 클러스터 **내부에서만 통신할** 수 있음. 보통 Deployment에 할당하고 서로 다른 Deployment나 StatefulSet과 통신이 필요할 때 사용.
  - 내부적으로 `endpoint` object로 IP 리스트를 관리함. 여러 개의 복제된 pod에 서비스를 연결하면 여러 개의 pod IP를 관리함.

![](/assets/img/docker/service.png){: width="400"}

- `NodePort`
  - ClusterIP는 내부 통신용이기 때문에 외부에서 접속하려면 각 노드에 외부에서 접속할 수 있는 port를 오픈해야 함
  - NodePort는 각 VM에 port를 오픈하고 들어온 요청을 내부의 ClusterIP로 연결함
  - NodePort를 만들면 ClusterIP가 자동으로 생성됨

![](/assets/img/docker/nodeport.png){: width="400"}

- `Loadbalancer`
  - NodePort를 외부에 연결된 LoadBalancer와 연결
  - `LoadBalancer` > `NodePort` > `ClusterIP` 순으로 연결됨

![](/assets/img/docker/loadbalancer.png){: width="400"}

- `Ingress`
  - Loadbalancer는 IP로만 생성되기 때문에 service마다 IP를 계속 만들어야 해서 비용 발생 → 하나의 IP를 사용하되 도메인 또는 Path에 따라 내부 ClusterIP와 연결하는 것이 Ingress (보통 모든 노드에 80/443 port로 오픈)
  - 기본 내장되어 있지 않고 상황에 따라 Ingress Controller를 pod으로 배포하여 사용할 수 있도록 유연한 구조로 설계되어 있음
  - nginx ingress, gce ingress 등

![](/assets/img/docker/ingress.png)

### 3.4. Storage

- 데이터를 저장하기 위해 기본적인 `Volume` 외에도 다양한 종류의 volume 지원
- `PersistentVolume` 관리자가 여러 개의 volume을 미리 생성해 두면 pod에서 volume을 요청할 때 조건(용량/라벨)에 맞는 volume을 사용
- `PersistentVolumeClaims` 관리자가 미리 volume을 생성하는 것이 아니라 pod을 생성할 때 동적으로 volume을 만드는 역할
- storage의 class(등급)을 지정해 둠

### 3.5. ConfigMap / Secret

- `ConfigMap` 공개되어도 괜찮은 설정 정보를 관리
- `Secret` 암호화하여 저장할 정보를 관리
- pod에서는 환경변수 또는 파일로 바인딩하여 사용 가능

## Reference

- [kubernetes architecture](https://medium.com/@kavishkafernando/exploring-the-kubernetes-architecture-a-foundation-for-modern-application-deployment-f2c0f15d661e)
- [쿠버네티스 시작하기](https://subicura.com/2019/05/19/kubernetes-basic-1.html)
- [Kubernetes NodePort vs LoadBalancer vs Ingress? When should I use what?](https://medium.com/google-cloud/kubernetes-nodeport-vs-loadbalancer-vs-ingress-when-should-i-use-what-922f010849e0)

