# D-Jobs_24_02
D-Jobs 아카데미 2기 파이썬 / 인공지능 강의 자료 

### Python 환경 세팅 

1. [아나콘다 설치 주소](https://www.anaconda.com/download)

    주의사항 : 파이썬이 이미 설치되어 있으신 분들은 반드시 삭제 후 아나콘다를 설치 해 주세요! 중복으로 설치할 경우 환경 변수 충돌 등으로 문제를 일으킬 수 있습니다! README.md 파일 정리

2. 아나콘다 사용법 
```
 01. 시작 메뉴에서 Anconda prompt 관리자 권한으로 실행 

 02. 가상 환경 생성 방법 
    conda create -n <환경명> python=<버전>
    예시) conda create -n AI python=3.11
    참고 : python, 인공지능 교육 동안 사용할 python veriosn = 3.11 입니다.

 03. conda 많이 사용한 명령어  
    ** 환경 생성 및 관리 ** 
    conda create --name <env_name>: 새로운 가상 환경 생성
    conda activate <env_name>: 가상 환경 활성화
    conda deactivate: 가상 환경 비활성화
    conda env list: 가상 환경 목록 표시
    conda env remove --name <env_name>: 가상 환경 삭제
    
    ** 패키지 설치 및 관리 ** 
    conda install <package_name>: 패키지 설치
    conda install -c <channel> <package_name>: 특정 채널에서 패키지 설치
    conda install --file <requirements.txt>: requirements.txt 파일에 명시된 패키지들을 한 번에 설치
    conda remove <package_name>: 패키지 제거
    conda update <package_name>: 패키지 업데이트
    conda list: 설치된 패키지 목록 표시
    
    ** 패키지 관리 및 환경 복제 ** 
    conda list --export > <environment_file>: 현재 환경의 패키지 목록을 내보내기
    conda create --name <new_env_name> --file <environment_file>: 다른 환경으로부터 패키지들을 포함한 새로운 환경 생성
    
    ** 기타 유틸리티 **
    conda info: 현재 설치된 conda에 대한 정보 표시
    conda search <package_name>: 패키지 검색
    conda config --set <option> <value>: conda 설정 변경 (예: channels, default_channels 등)
```
3. [PyChame 설치 주소](https://www.jetbrains.com/ko-kr/pycharm/download/download-thanks.html?platform=windows&code=PCC)

