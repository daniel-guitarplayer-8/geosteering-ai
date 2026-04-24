module utils
  use parameters
contains
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine sanitize_hprof_well(n, esp, h, prof)
  ! Input:
  ! n: número de camadas
  ! esp: array com as espessuras das n-1 camadas.
  ! Output:
  ! h: array de dimensão n+1 com as espessuras de cada camada (índices de 1 a n-1), mais h(0) = 0 e h(n)=1.d300, valores que
  ! contornam problemas em valores limites nas exponenciais);
  ! prof: array de dimensão n+2 com as espessuras das n-1 camadas (índices de 1 a n-1), mais prof(-1)=-1.d300, prof(0)=0.d0 e  
  ! prof(n) = 1.d300, os quais são valores que contornam a problemática de convergência em avaliações de expoenciais.
  implicit none
  integer :: n
  real(dp), intent(in) ::esp(:)
  real(dp), dimension(:), allocatable, intent(out) :: h, prof

  integer :: k  !i, j, 

  allocate( h(1:n), prof(0:n) )
  if (size(esp) == n) then
    h(1) = 0.d0
    h(2:n-1) = esp(2:n-1)
    h(n) = 0.d0
  else
    h(1) = 0.d0
    h(2:n-1) = esp
    h(n) = 0.d0 !1.d300
  end if
  ! create depths array that suits any pathological situation
  prof(0) = -1.d300
  if (n > 1) then
    prof(1) = h(1)
    if (n > 2) then
      do k = 2, n-1
        prof(k) = prof(k-1) + h(k)
      end do
    end if
  end if
  prof(n)=1.d300

end subroutine sanitize_hprof_well
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine findlayersTR2well(n, h0, z, prof, camadT, camad)
  ! Input:
  ! n: número de camadas
  ! h0: altura (valor negativo) ou profundidade (valor positivo) do transmissor em relação à 1ª interface (ar-primeira camada);
  ! z: profundidade de medida (posição do receptor);
  ! prof: array com as profundidades das n-1 camadas, com input sendo de 1 até n-1.
  ! Output:
  ! camadT: camada onde está localizado o transmissor;
  ! camad: camada onde está o receptor.
  implicit none
  integer :: n
  real(dp), intent(in) :: h0, z, prof(1:n-1)
  integer, intent(out) :: camadT, camad

  integer :: i, j !, k

  ! find the layer where the receiver is
  camad = 1 !when z = 0, receiver in the first layer is preferable to avoid inaccuracy (em componentes não tangenciais)
  if (z >= prof(n-1)) then
    camad = n
  else
    do i = n-1, 2, -1
      if (z >= prof(i-1)) then
        camad = i
        exit
      end if
    end do
  end if

  ! find the layer where the transmitter is
  camadT = 1  !when h0=0, transmitter in the air is preferable to avoid inaccuracy (em componentes não tangenciais)
  if (h0 > prof(n-1)) then
    camadT = n
  else
    do j = n-1, 2, -1
      if (h0 > prof(j-1)) then
        camadT = j
        exit
      end if
    end do
  end if

end subroutine findlayersTR2well
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine sanitizedata2well(n, h0, z, esp, camadT, camad, h, prof)
  ! Input:
  ! n: número de camadas
  ! h0: altura (valor negativo) ou profundidade (valor positivo) do transmissor em relação à 1ª interface (ar-primeira camada);
  ! z: profundidade de medida (posição do receptor);
  ! esp: array com as espessuras das n-1 camadas.
  ! Output:
  ! camadT: camada onde está localizado o transmissor;
  ! camad: camada onde está o receptor;
  ! h: array de dimensão n+1 com as espessuras de cada camada (índices de 1 a n-1), mais h(0) = 0 e h(n)=1.d300, valores que
  ! contornam problemas em valores limites nas exponenciais);
  ! prof: array de dimensão n+2 com as espessuras das n-1 camadas (índices de 1 a n-1), mais prof(-1)=-1.d300, prof(0)=0.d0 e  
  ! prof(n) = 1.d300, os quais são valores que contornam a problemática de convergência em avaliações de expoenciais.
  implicit none
  integer :: n
  real(dp), intent(in) :: h0, z, esp(:)
  integer, intent(out) :: camadT, camad
  real(dp), dimension(:), allocatable, intent(out) :: h, prof

  integer :: i, j, k

  allocate( h(1:n), prof(0:n) )
  if (size(esp) == n) then
    h(1) = 0.d0
    h(2:n) = esp(2:n)
  else
    h(1) = 0.d0
    h(2:n-1) = esp
    h(n) = 1.d300
  end if
  ! create depths array that suits any pathological situation
  prof(0) = -1.d300
  if (n > 1) then
    prof(1) = h(1)
    if (n > 2) then
      do k = 2, n-1
        prof(k) = prof(k-1) + h(k)
      end do
    end if
  end if
  prof(n)=1.d300

  ! find the layer where the receiver is
  camad = 1 !when z = 0, receiver in the first layer is preferable to avoid inaccuracy (em componentes não tangenciais)
  if (z >= prof(n-1)) then
    camad = n
  else
    do i = n-1, 2, -1
      if (z >= prof(i-1)) then
        camad = i
        exit
      end if
    end do
  end if

  ! find the layer where the transmitter is
  camadT = 1  !when h0=0, transmitter in the air is preferable to avoid inaccuracy (em componentes não tangenciais)
  if (h0 > prof(n-1)) then
    camadT = n
  else
    do j = n-1, 2, -1
      if (h0 > prof(j-1)) then
        camadT = j
        exit
      end if
    end do
  end if
end subroutine sanitizedata2well
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine commonarraysMD(n, npt, hordist, krJ0J1, zeta, h, eta, u, s, uh, sh, &
                          RTEdw, RTEup, RTMdw, RTMup, AdmInt)
  use parameters
  ! Versão para anisotropia TIV
    ! INPUT:
    ! n: número de camadas
    ! npt: número de pontos do filtro
    ! hordist: distância horizontal entre o transmissor e o receptor, i.e, r
    ! krJ0J1: array com as abscissas do filtro das transformadas de Hankel
    ! zeta: impeditividade (i\omega\mu)
    ! h0: altura (negativo, quando no ar), ou profundidade de posição do transmissor, em relação à origem (1ª interface ar-terra)
    ! h: array das espessuras das n-1 primeiras camadas
    ! prof: array com as profundidades das interfaces das camadas (é calculada em sanitizedata)
    ! camadT: inteiro designando em que camada está o transmissor
    ! eta: array 2D com a primeira coluna sendo a condutividade horizontal e a segunda a condutividade vertical, de cada camada
    ! OUTPUT:
    ! u: constante de propagação horizontal de cada camada
    ! s: constante de propagação vertical de cada camada, multiplicada por lambda (\sqrt{\sigma_h/\sigma_v}). I.e.: lambda*v
    ! lamb2: razão da condutividade horizontal pela condutividade vertical, ao quadrado (\sigma_h/\sigma_v) de cada camada.
    ! AdmInt: admitância intríseca de cada camada
    ! ImpInt: impedância intríseca de cada camada
    ! RTEdw: coeficientes de reflexão das camadas inferiores do modo TE
    ! RTEup: coeficientes de reflexão das camadas superiores do modo TE
    ! RTMdw: coeficientes de reflexão das camadas inferiores do modo TM
    ! RTMup: coeficientes de reflexão das camadas superiores do modo TM
  implicit none
  integer, intent(in) :: n, npt
  real(dp), intent(in) :: hordist, krJ0J1(npt), h(1:n), eta(n,2)
  complex(dp), intent(in) :: zeta
  complex(dp), dimension(npt,1:n), intent(out) :: u, s, uh, sh, RTEdw, RTMdw, RTEup, RTMup, AdmInt

  integer :: i
  real(dp) :: r, kr(npt), lamb2(1:n)
  complex(dp) :: kh2(1:n), kv2(1:n), tghuh(npt,1:n), tghsh(npt,1:n), v(npt,1:n)
  complex(dp) :: ImpInt(npt,1:n), AdmApdw(npt,1:n), ImpApdw(npt,1:n), AdmApup(npt,1:n), ImpApup(npt,1:n)

  if (hordist < eps) then
    r = 1.d-2 !valor usado no caso da singularidade
  else
    r = hordist
  end if

  kr = krJ0J1 / r
  do i = 1, n
    kh2(i) = -zeta * eta(i,1)
    kv2(i) = -zeta * eta(i,2)
    lamb2(i) = eta(i,1) / eta(i,2)
    u(:,i) = sqrt(kr * kr - kh2(i))
    v(:,i) = sqrt(kr * kr - kv2(i))
    s(:,i) = sqrt(lamb2(i)) * v(:,i)
    AdmInt(:,i) = u(:,i) / zeta
    ImpInt(:,i) = s(:,i) / eta(i,1)
    uh(:,i) = u(:,i) * h(i)
    sh(:,i) = s(:,i) * h(i)
    tghuh(:,i) = (1.d0 - exp(-2.d0 * uh(:,i))) / (1.d0 + exp(-2.d0 * uh(:,i)))
    tghsh(:,i) = (1.d0 - exp(-2.d0 * sh(:,i))) / (1.d0 + exp(-2.d0 * sh(:,i)))
  end do

  AdmApdw(:,n) = AdmInt(:,n)
  ImpApdw(:,n) = ImpInt(:,n)
  RTEdw(:,n) = (0.d0,0.d0)
  RTMdw(:,n) = (0.d0,0.d0)
  do i = n-1, 1, -1
    AdmApdw(:,i) = AdmInt(:,i) * (AdmApdw(:,i+1) + AdmInt(:,i) * &
                  tghuh(:,i)) / (AdmInt(:,i) + AdmApdw(:,i+1) * tghuh(:,i))
    ImpApdw(:,i) = ImpInt(:,i) * (ImpApdw(:,i+1) + ImpInt(:,i) * &
                  tghsh(:,i)) / (ImpInt(:,i) + ImpApdw(:,i+1) * tghsh(:,i))
    RTEdw(:,i) = (AdmInt(:,i) - AdmApdw(:,i+1)) / (AdmInt(:,i) + AdmApdw(:,i+1))
    RTMdw(:,i) = (ImpInt(:,i) - ImpApdw(:,i+1)) / (ImpInt(:,i) + ImpApdw(:,i+1))
  end do

  AdmApup(:,1) = AdmInt(:,1)
  ImpApup(:,1) = ImpInt(:,1)
  RTEup(:,1) = (0.d0,0.d0)
  RTMup(:,1) = (0.d0,0.d0)
  do i = 2, n
    AdmApup(:,i) = AdmInt(:,i) * (AdmApup(:,i-1) + AdmInt(:,i) * &
                tghuh(:,i)) / (AdmInt(:,i) + AdmApup(:,i-1) * tghuh(:,i))
    ImpApup(:,i) = ImpInt(:,i) * (ImpApup(:,i-1) + ImpInt(:,i) * &
                tghsh(:,i)) / (ImpInt(:,i) + ImpApup(:,i-1) * tghsh(:,i))
    RTEup(:,i) = (AdmInt(:,i) - AdmApup(:,i-1)) / (AdmInt(:,i) + AdmApup(:,i-1))
    RTMup(:,i) = (ImpInt(:,i) - ImpApup(:,i-1)) / (ImpInt(:,i) + ImpApup(:,i-1))
  end do
end subroutine commonarraysMD
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
subroutine commonfactorsMD(n, npt, h0, h, prof, camadT, u, s, uh, sh, RTEdw, RTEup, RTMdw, RTMup, Mxdw, Mxup, &
            Eudw, Euup, FEdwz, FEupz)
  ! Subrotina de cálculo de fatores de onda refletida da camada onde está o transmissor. Útil como etapa de otimização, 
    ! pois, quando a distância horizontal r entre T e R é a mesma de uma posição T-R anterior, não mais precisamos chamar
    ! a subrotina commonarraysMD (e que possui muitos arrays necessários para a modelagem), tendo ainda as constantes aqui
    ! calculadas como sendo outras comuns, mas alterando-se quando a camada onde T está, é diferente da investigação anterior.
  ! INPUT:
    ! npt: número de pontos do filtro
    ! h0: altura (negativo, quando no ar), ou profundidade de posição do transmissor, em relação à origem (1ª interface ar-terra)
    ! h: array das espessuras das n-1 primeiras camadas
    ! prof: array com as profundidades das interfaces das camadas (é calculada em sanitizedata)
    ! camadT: inteiro designando em que camada está o transmissor
    ! u: constante de propagação horizontal de cada camada
    ! s: constante de propagação vertical de cada camada, multiplicada por lambda (\sqrt{\sigma_h/\sigma_v}). I.e.: lambda*v
    ! uh: produtos de u por h, de cada camada
    ! sh: produtos de s por h, de cada camada
    ! RTEdw: coeficiente de reflexão das interfaces inferiores do modo TE
    ! RTEup: coeficiente de reflexão das interfaces superiores do modo TE
    ! RTMdw: coeficiente de reflexão das interfaces inferiores do modo TM
    ! RTMup: coeficiente de reflexão das interfaces superiores do modo TM
  ! Output:
    ! Mxdw: fator de onda refletida pelas camadas inferiores do modo TE (potencial \pi_x)
    ! Mxup: fator de onda refletida pelas camadas superiores do modo TE
    ! Eudw: fator de onda refletida pelas camadas inferiores do modo TM (potencial \pi_u)
    ! Euup: fator de onda refletida pelas camadas superiores do modo TM
  implicit none
  integer, intent(in) :: npt, n, camadT
  real(dp), intent(in) :: h0, h(1:n), prof(0:n)
  complex(dp), dimension(npt,1:n), intent(in) :: u, s, uh, sh, RTEdw, RTMdw, RTEup, RTMup
  complex(dp), dimension(npt), intent(out) :: Mxdw, Mxup, Eudw, EUup, FEdwz, FEupz

  complex(dp) :: den(npt)

  ! Rx, Tx e Imp estão associados ao modo TM, enquanto Ru, Tu e Adm estão associados ao modo TE
  den = 1.d0 - RTMdw(:,camadT) * RTMup(:,camadT) * exp(-2.d0 * sh(:,camadT))
  Mxdw = (exp(-s(:,camadT) * (prof(camadT) - h0)) + RTMup(:,camadT) * &
          exp(s(:,camadT) * (prof(camadT-1) - h0 - h(camadT)))) / den

  Mxup = (exp(s(:,camadT) * (prof(camadT-1) - h0)) + RTMdw(:,camadT) * &
          exp(-s(:,camadT) * (prof(camadT) - h0 + h(camadT)))) / den

  den = 1.d0 - RTEdw(:,camadT) * RTEup(:,camadT) * exp(-2.d0 * uh(:,camadT))
  Eudw = (exp(-u(:,camadT) * (prof(camadT) - h0)) - RTEup(:,camadT) * &
          exp(u(:,camadT) * (prof(camadT-1) - h0 - h(camadT)))) / den

  Euup = (exp(u(:,camadT) * (prof(camadT-1) - h0)) - RTEdw(:,camadT) * &
          exp(-u(:,camadT) * (prof(camadT) - h0 + h(camadT)))) / den
  
  ! Usados na modelagem com o DMV:
  FEdwz = (exp(-u(:,camadT) * (prof(camadT)-h0)) + RTEup(:,camadT) * &
          exp(u(:,camadT) * (prof(camadT-1) - h(camadT) - h0))) / den

  FEupz = (exp(u(:,camadT) * (prof(camadT-1)-h0)) + RTEdw(:,camadT) * &
          exp(-u(:,camadT) * (prof(camadT) + h(camadT) - h0))) / den
end subroutine commonfactorsMD
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
function layer2z_inwell(n, z, profs)  !camada em que z se encontra
  implicit none
  integer, intent(in) :: n
  real(dp), intent(in) :: z, profs(1:n-1)
  integer :: layer2z_inwell

  integer :: i

  ! encontra a camada na qual uma cota z está
  layer2z_inwell = 1
  if (z >= profs(n-1)) then
    layer2z_inwell = n
  else
    do i = n-1, 2, -1
      if (z >= profs(i-1)) then
        layer2z_inwell = i
        exit
      end if
    end do
  end if
end function layer2z_inwell
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
function RtHR(alpha, beta, gamma, H)  !Campo magnético com eixo da triaxial em orientação arbitrária
  !Liu (2017); Theory of Electromagnetic Well Logging, página 148, equação 4.80 e figura 4.17
    ! Input:
      ! alpha: ângulo que o segmento (que vai da observação até a origem) faz com o eixo OZ
      ! beta: ângulo que o segmento projeção (da observação no plano XOY) faz com o eixo OX
      ! gamma: ângulo de rotação em torno do eixo da ferramenta
      ! H: matriz sinal de indução magnética, tendo cada linha o campo decorrente do dipolo na direção
      !                    1 <==> x
      !                    2 <==> y
      !                    3 <==> z
      ! e, cada coluna representado o valor do campo para a direção
      !                    1 <==> x
      !                    2 <==> y
      !                    3 <==> z
  implicit none
  real(dp), intent(in) :: alpha, beta, gamma
  complex(dp), intent(in) :: H(3,3)
  complex(dp) :: RtHR(3,3)

  real(dp) :: sena, cosa, senb, cosb, seng, cosg, R(3,3), Rt(3,3)

  sena = sin(alpha)
  cosa = cos(alpha)
  senb = sin(beta)
  cosb = cos(beta)
  seng = sin(gamma)
  cosg = cos(gamma)

  R(1,:) = (/cosa * cosb * cosg - senb * seng, -cosa * cosb * seng - senb * cosg, sena * cosb/)
  R(2,:) = (/cosa * senb * cosg + cosb * seng, -cosa * senb * seng + cosb * cosg, sena * senb/)
  R(3,:) = (/ -sena * cosg, sena * seng, cosa/)
  Rt = transpose(R)
  RtHR = matmul(matmul(Rt,H),R)

end function RtHR
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
function int2str(num)
  implicit none
  integer, intent(in) :: num
  character(len=16) :: int2str
  write(int2str, '(i1)') num
end function int2str
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
function real2str(real, format)
  implicit none
  real(sp), intent(in) :: real
  character(len=*), intent(in), optional :: format
  character(len=16) :: real2str
  if (present(format)) then
    write(real2str, format) real
  else
    write(real2str, '(f6.2)') real
  end if
end function real2str
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
function real2strPerc(real)
  implicit none
  real(sp), intent(in) :: real
  character(len=3) :: real2strPerc
  write(real2strPerc, '(i3)') nint(real)
end function real2strPerc
!§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§
end module utils
