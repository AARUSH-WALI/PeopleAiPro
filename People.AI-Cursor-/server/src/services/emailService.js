import nodemailer from 'nodemailer';
import dotenv from 'dotenv';

dotenv.config();

// Create a Nodemailer transporter using SMTP settings from environment
const createSmtpTransporter = () => {
  const host = (process.env.SMTP_SERVER || 'smtp.gmail.com').trim();
  const port = Number((process.env.SMTP_PORT || 587).toString().trim());
  const rawUser = (process.env.EMAIL_ADDRESS || process.env.GMAIL_USER || '').trim();
  const rawPass = (process.env.EMAIL_PASSWORD || process.env.GMAIL_APP_PASSWORD || '').trim();
  const user = rawUser;
  const pass = rawPass.replace(/\s+/g, ''); // remove spaces Google shows in app passwords

  if (!user || !pass) {
    throw new Error('Email credentials are not configured (set EMAIL_ADDRESS/EMAIL_PASSWORD or GMAIL_USER/GMAIL_APP_PASSWORD)');
  }

  const secure = port === 465; // true for port 465, false otherwise (e.g., 587)

  const transporter = nodemailer.createTransport({
    host,
    port,
    secure,
    auth: { user, pass },
  });

  return transporter;
};

/**
 * Send personality test invitation email to candidate
 * @param {string} to - Recipient email address
 * @param {string} candidateName - Candidate's name
 * @param {string} testLink - URL to the Big5 personality test
 * @returns {Promise<{success: boolean, messageId?: string, error?: string}>}
 */
export async function sendPersonalityTestEmail(to, candidateName, testLink) {
  try {
    if (!to || !to.trim()) {
      throw new Error('Recipient email is required');
    }

    const defaultFrom = process.env.EMAIL_ADDRESS || process.env.GMAIL_USER;
    const fromEmail = process.env.EMAIL_FROM || defaultFrom;

    // Create email HTML template (modern, responsive, accessible)
    const emailHtml = `
      <!DOCTYPE html>
      <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <meta name="x-apple-disable-message-reformatting">
          <title>You're invited to take your Personality Assessment</title>
          <style>
            @media (max-width: 600px) {
              .container { padding: 16px !important; }
              .card { padding: 20px !important; }
              .cta { width: 100% !important; display: block !important; }
            }
          </style>
        </head>
        <body style="margin:0; padding:0; background:#f8fafc; -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;">
          <div role="article" aria-roledescription="email" aria-label="Personality Assessment Invitation" lang="en">
            <div class="preheader" style="display:none;overflow:hidden;line-height:1px;opacity:0;max-height:0;max-width:0;color:transparent;">
              Your personalized personality assessment is ready. It only takes 10–15 minutes.
            </div>
            <table align="center" width="100%" border="0" cellPadding="0" cellSpacing="0" role="presentation" style="background:#f8fafc;">
              <tr>
                <td>
                  <div class="container" style="max-width:640px;margin:0 auto;padding:24px;">
                    <table width="100%" role="presentation" style="border-collapse:separate; border-spacing:0;">
                      <tr>
                        <td style="padding:0;">
                          <div style="background:linear-gradient(135deg,#fb923c 0%,#f59e0b 40%,#facc15 100%);border-radius:16px 16px 0 0;padding:32px;text-align:center;box-shadow:0 10px 20px rgba(245, 158, 11, 0.25);">
                            <h1 style="margin:0;font-family:Segoe UI, Roboto, Arial, sans-serif;color:#0b1324;font-size:28px;letter-spacing:-0.02em;">
                              Personality Assessment Invitation
                            </h1>
                            <p style="margin:8px 0 0 0;color:#0b1324;opacity:.8;font-size:14px;">
                              From People AI • Tailored insights to help you shine
                            </p>
                          </div>
                          <div class="card" style="background:#ffffff;padding:28px;border:1px solid #e5e7eb;border-top:none;border-radius:0 0 16px 16px;">
                            <p style="font:16px/1.6 Segoe UI, Roboto, Arial, sans-serif;color:#0b1324;margin:0 0 14px 0;">Hello ${candidateName || 'Candidate'},</p>
                            <p style="font:16px/1.7 Segoe UI, Roboto, Arial, sans-serif;color:#334155;margin:0 0 16px 0;">
                              Thank you for your interest in joining our team. We'd love to learn more about your strengths and work style through a brief personality assessment.
                            </p>
                            <ul style="padding-left:18px;margin:0 0 18px 0;color:#475569;font:15px/1.7 Segoe UI, Roboto, Arial, sans-serif;">
                              <li>Only 10–15 minutes to complete</li>
                              <li>50 thoughtfully designed questions</li>
                              <li>Helps us match you with the right role</li>
                            </ul>
                            <div style="text-align:center;margin:28px 0;">
                              <a href="${testLink}" class="cta" 
                                style="background:linear-gradient(135deg,#f59e0b 0%,#fb923c 100%);color:#0b1324;text-decoration:none;padding:14px 28px;border-radius:10px;font-weight:700;font-size:16px;display:inline-block;border:1px solid rgba(11,19,36,.08);box-shadow:0 8px 16px rgba(245, 158, 11, 0.25);">
                                Start your assessment
                              </a>
                              <div style="font-size:12px;color:#64748b;margin-top:10px;">Secure link • No sign‑in required</div>
                            </div>
                            <div style="background:#fff7ed;border:1px solid #ffedd5;padding:16px;border-radius:12px;">
                              <p style="margin:0;color:#7c2d12;font:14px/1.6 Segoe UI, Roboto, Arial, sans-serif;">
                                <strong>Friendly reminder:</strong> Please complete the assessment within 7 days. Your responses are private and used only for recruitment.
                              </p>
                            </div>
                            <div style="margin-top:22px;border-top:1px dashed #e5e7eb;padding-top:18px;color:#475569;font:14px/1.7 Segoe UI, Roboto, Arial, sans-serif;">
                              <p style="margin:0 0 8px 0;">If you have any questions, just reply to this email—we’re here to help.</p>
                              <p style="margin:0;color:#0b1324;">Warm regards,<br><strong>The People AI Team</strong></p>
                            </div>
                          </div>
                          <div style="text-align:center;margin-top:16px;padding:12px;color:#94a3b8;font:12px/1.6 Segoe UI, Roboto, Arial, sans-serif;">
                            <p style="margin:0;">You’re receiving this because you applied to a role with us.</p>
                            <p style="margin:0;">© ${new Date().getFullYear()} People AI • All rights reserved</p>
                          </div>
                        </td>
                      </tr>
                    </table>
                  </div>
                </td>
              </tr>
            </table>
          </div>
        </body>
      </html>
    `;

    // Plain text version
    const emailText = `
Hello ${candidateName || 'Candidate'},

Thank you for applying with us! As part of our recruitment process, we'd like to invite you to complete a brief personality assessment.

This assessment helps us better understand your work style and find the best role fit for you. It takes approximately 10-15 minutes to complete and consists of 50 questions.

Take the test here: ${testLink}

Note: Please complete this assessment within the next 7 days. Your responses are confidential and will be used solely for recruitment purposes.

If you have any questions or concerns, please don't hesitate to reach out to us.

Best regards,
The Recruitment Team
    `.trim();
    console.log('Sending personality test email to:', to);
    const transporter = createSmtpTransporter();
    const smtpInfo = {
      host: process.env.SMTP_SERVER || 'smtp.gmail.com',
      port: Number(process.env.SMTP_PORT || 587),
      user: (process.env.EMAIL_ADDRESS || process.env.GMAIL_USER || '').replace(/.(?=.{3})/g, '*'),
    };
    console.log('SMTP config:', smtpInfo);
    try {
      await transporter.verify();
    } catch (verifyErr) {
      console.error('SMTP verification failed:', {
        message: verifyErr.message,
        code: verifyErr.code,
      });
      throw new Error(`SMTP connection failed: ${verifyErr.message}`);
    }
    const info = await transporter.sendMail({
      from: fromEmail,
      to: to.trim(),
      subject: 'Personality Assessment Invitation - Complete Your Test',
      html: emailHtml,
      text: emailText,
    });

    console.log('Email sent successfully:', info.messageId);
    return { success: true, messageId: info.messageId, message: 'Email sent successfully' };
  } catch (error) {
    console.error('Error sending email:', {
      message: error.message,
      code: error.code,
      stack: error.stack,
    });
    return {
      success: false,
      error: error.message || 'An unexpected error occurred while sending email'
    };
  }
}

/**
 * Send general email (for future use)
 * @param {string} to - Recipient email
 * @param {string} subject - Email subject
 * @param {string} html - HTML content
 * @param {string} text - Plain text content
 */
export async function sendEmail(to, subject, html, text) {
  try {
    const transporter = createSmtpTransporter();
    try {
      await transporter.verify();
    } catch (verifyErr) {
      console.error('SMTP verification failed:', {
        message: verifyErr.message,
        code: verifyErr.code,
      });
      throw new Error(`SMTP connection failed: ${verifyErr.message}`);
    }
    const defaultFrom = process.env.EMAIL_ADDRESS || process.env.GMAIL_USER;
    const fromEmail = process.env.EMAIL_FROM || defaultFrom;

    const info = await transporter.sendMail({
      from: fromEmail,
      to: to.trim(),
      subject,
      html,
      text,
    });

    return { success: true, messageId: info.messageId };
  } catch (error) {
    console.error('Error sending email:', {
      message: error.message,
      code: error.code,
      stack: error.stack,
    });
    return {
      success: false,
      error: error.message
    };
  }
}

