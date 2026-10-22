% Better ADSAI created by Jordy Slagter, original by Zuyd University
% https://github.com/jordyslagter/better_adsai

\ProvidesClass{adsai}

\RequirePackage{kvoptions}

\SetupKeyvalOptions{
  family=adsai,
  prefix=adsai@
}

\DeclareStringOption[ieee]{bibstyle}
\DeclareBoolOption[false]{dutch}

\DeclareVoidOption*{
  \PassOptionsToClass{\CurrentOption}{article}%
}

\ProcessKeyvalOptions*

\LoadClass[10pt, a4paper]{article}

\RequirePackage[a4paper, heightrounded]{geometry}
\ifadsai@dutch
  \RequirePackage[dutch]{babel}
\else
  \RequirePackage[english]{babel}
\fi
\RequirePackage[parfill]{parskip}
\RequirePackage{graphicx}
\RequirePackage{wrapfig}
\RequirePackage{amsmath, amsfonts}
\RequirePackage{enumerate}
\RequirePackage[most]{tcolorbox}
\RequirePackage{hyperref}
\RequirePackage{float}
\RequirePackage{dsfont}
\RequirePackage{caption}
\RequirePackage{tabularx}
\RequirePackage{booktabs} % for professional tables
\RequirePackage{tikz}
\usetikzlibrary{trees}
\RequirePackage{svg}
\RequirePackage{subcaption}
\RequirePackage{xurl}
\RequirePackage{placeins}
\RequirePackage{bm}
\RequirePackage{listings}
\RequirePackage{layout}
\RequirePackage{enumitem}
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}
\RequirePackage{appendix}
\RequirePackage{listings}
\RequirePackage{eqparbox}
\RequirePackage{sourcecodepro}

\RequirePackage{csquotes}
\RequirePackage[style=\adsai@bibstyle]{biblatex}
\addbibresource{references.bib}

\geometry{vmargin=2cm,hmargin=3cm}

\definecolor{codegreen}{rgb}{0,0.5,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.92}
\definecolor{backcolour}{rgb}{.98,.97,.95}

\lstset{
  language={Python},
  backgroundcolor=\color{backcolour},
  commentstyle=\color[rgb]{0.13,0.54,0.13},
  keywordstyle=\color{blue},
  numberstyle=\tiny\color{codegray},
  stringstyle=\color{codepurple},
  basicstyle=\ttfamily\footnotesize,
  breakatwhitespace=false,
  breaklines=true,
  captionpos=b,
  keepspaces=true,
  numbers=left,
  numbersep=10pt,
  showspaces=false,
  showstringspaces=false,
  showtabs=false,
  tabsize=2,
  keywordsprefix=\#,alsoletter=\#,%
  morekeywords={
    Console,
    WriteLine,
    int,
  },
}

\lstnewenvironment{python}[2]
{
  \lstset{
    caption=#1,
    label=#2
  }%
}{%
}

\algtext*{EndWhile}% Remove "end while" text
\algtext*{EndFor}% Remove "end while" text
\algtext*{EndIf}% Remove "end if" text
\algtext*{EndFunction}% Remove "end if" text

\newcommand{\HRule}[1]{\rule{\linewidth}{#1}}

% Change caption titles to Dutch
\ifadsai@dutch
  %\renewcommand{\tablename}{Tabel}
  %\renewcommand{\figurename}{Figuur}
  \renewcommand{\lstlistingname}{Code}
  \floatname{algorithm}{Algoritme}
\fi

\renewcommand{\@title}{}
\renewcommand{\title}[1]{
  \gdef\@title{
    \normalsize\textsc{}\\[2.0cm]
    \HRule{1.5pt}\\
    \LARGE\textbf{\MakeUppercase{Extended Abstract}}\\
    \LARGE\textbf{\MakeUppercase{#1}}\\
    \HRule{2.0pt}\\[0.6cm]
    \LARGE
    \@modulename\\
    Lectoraat Data Intelligence\\
    Zuyd Hogeschool
  }
}

\newcommand{\@academicyear}{}
\newcommand{\academicyear}[1]{\gdef\@academicyear{#1}}

\newcommand{\@period}{}
\newcommand{\period}[1]{\gdef\@period{#1}}

\newcommand{\@modulename}{}
\newcommand{\modulename}[1]{\gdef\@modulename{#1}}

\renewcommand{\@date}{}
\renewcommand{\date}[1]{\gdef\@date{
  \@academicyear, \@period \\
  #1
}}

\captionsetup[table]{skip=10pt}

\let\oldabstract\abstract
\renewcommand{\abstract}[1]{
  \ifadsai@dutch
    \renewcommand{\abstractname}{Abstract}
  \fi
  \oldabstract{#1}
  \thispagestyle{empty}
}

\let\oldprintbibliography\printbibliography
\renewcommand{\printbibliography}{
  \ifadsai@dutch
    \renewcommand{\refname}{Verwijzingen}
  \fi

  \oldprintbibliography
}

\let\oldappendixpage\appendixpage
\renewcommand{\appendixpage}{
  \ifadsai@dutch
    \renewcommand{\appendixpagename}{Bijlagen}
  \fi

  \oldappendixpage
}

% set up book-style alignment, has to be here otherwise local fonts won't work
% anymore
\RequirePackage{microtype}
% prefer stretching over overfilling
\setlength{\emergencystretch}{2em}
\tolerance=1000
% allow the end of a paragraph to be left aligned
\setlength{\parfillskip}{0pt plus 2fil}

\endinput

