import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Sans_Thai, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["400", "600", "700"],
});

const plexThai = IBM_Plex_Sans_Thai({
  variable: "--font-plex-thai",
  subsets: ["latin", "thai"],
  weight: ["400", "600"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "Thai Cybersecurity RAG — Evidence Console",
  description:
    "Grounded, page-cited answers over OWASP Top 10, MITRE ATT&CK, and the Thailand Web Security Standard 2025.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${plexSans.variable} ${plexThai.variable} ${plexMono.variable}`}
    >
      <body>{children}</body>
    </html>
  );
}
